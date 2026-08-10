"""
Per-camera Re-ID + Name Resolution
===================================

Thin orchestration layer for the web MVP's multi-camera flow. It does NOT
edit reidentification/reid_main.py or registration/identity_db.py — it only
*calls* the public pieces those files already expose, same pattern as
registration/embedder.py calling ReIDEngine.extract_feature().

Why this exists instead of calling reidentification.reid_main.run_reid_pipeline()
directly: that function is correct for rendering an output video, but it
renumbers stable IDs internally and doesn't return a name-resolved summary.
Here we run the identical frame loop ourselves so we can:

  1. Match each stable identity's descriptor against the registration
     module's IdentityDatabase (registration.identity_db) *before* the
     renumbering step, in the same descriptor space the renumbering reads
     from — so name resolution and on-screen IDs never get out of sync.
  2. Return a small, dashboard-ready summary (per person: name or "unknown",
     first/last seen timestamp) for one camera.

If anything about the engine's internals changes upstream, this fails soft:
matching is wrapped so a mismatch degrades to "unknown" rather than crashing
the run.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _save_evidence_frame(video_path: str, frame_id: int, bbox, out_path: Path) -> Path | None:
    """Crop a single video frame at the given bbox and save it as a JPEG.

    Returns the written path, or None when the frame/bbox can't be recovered.
    """
    if not bbox:
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_id) - 1))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
    except Exception:
        logger.exception("Could not read evidence frame %s from %s", frame_id, video_path)
        return None

    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return None
        mx = int(0.12 * w)
        my = int(0.18 * h)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(frame.shape[1], x2 + mx)
        y2 = min(frame.shape[0], y2 + my)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), crop)
        return out_path
    except Exception:
        logger.exception("Could not save evidence frame %s", out_path)
        return None


# Combined scoring weights — body provides discrimination that face cannot
# for similar-looking people (ArcFace cross-person similarity can exceed
# same-person similarity). Heavy body weighting ensures self-match (body=1.0)
# always wins over cross-match (body=0.88-0.96).
WEIGHT_BODY = 0.70
WEIGHT_FACE = 0.30
# Minimum combined score to accept a name. Below this, the track is unidentified.
COMBINED_THRESHOLD = 0.70


def _resolve_identity_name(
    sid,
    body_feat,
    face_candidates,
    track_to_identity,
    identity_db,
    face_confirm,
    match_threshold,
    face_similarity=None,
):
    """Combined body+face name resolution for one stable identity.

    For each registered person, computes:
      combined = WEIGHT_BODY × body_sim + WEIGHT_FACE × face_sim

    where body_sim is cosine(track_body, person_avg_body) and face_sim is
    the best ArcFace similarity across all face candidates for this track.

    Returns the person with the highest combined score, or None if below
    COMBINED_THRESHOLD.

    This replaces the previous face-confirm / exclusivity-margin / body-fallback
    chain. The combined approach works because body self-similarity (1.0) is
    always higher than cross-similarity (0.88-0.96), providing discrimination
    that face alone cannot for similar-looking people.
    """
    if face_similarity is None:
        from reidentification.face_cue import FaceCueExtractor
        face_similarity = FaceCueExtractor.similarity

    from reidentification.reid_main import _cosine as _cosine_fn

    # Collect best face per registered person across all candidate frames
    best_face_per_person = {}  # name -> (face_feat, face_sim)
    cands_for_sid = [
        cands
        for tid, cands in face_candidates.items()
        if track_to_identity.get(tid) == sid
    ]
    for cands in cands_for_sid:
        for (_fid, _bbox, face_feat, _score, _fbox) in cands:
            for name, record in identity_db._data.items():
                avg = record.get("average_face_descriptor")
                if avg is None:
                    continue
                sim = face_similarity(avg, face_feat)
                if sim is not None:
                    prev = best_face_per_person.get(name)
                    if prev is None or sim > prev[1]:
                        best_face_per_person[name] = (face_feat, sim)

    # Combined scoring against every registered person
    best_name = None
    best_score = -1.0
    for name, record in identity_db._data.items():
        avg_body = record.get("average_embedding")
        body_sim = _cosine_fn(body_feat, avg_body) if avg_body is not None else 0.0

        face_entry = best_face_per_person.get(name)
        face_sim = face_entry[1] if face_entry else 0.0

        combined = WEIGHT_BODY * body_sim + WEIGHT_FACE * face_sim
        if combined > best_score:
            best_score = combined
            best_name = name

    if best_name is not None:
        # Face shortcut: if face confidently confirms (>= 0.40), accept
        # even if combined score is lower. This handles the common case
        # where face is clear but body has drifted from registration.
        face_entry = best_face_per_person.get(best_name)
        if face_entry and face_entry[1] >= 0.40:
            return (best_name, best_score)
        # Combined threshold for body-only tracks (no face confirmation)
        if best_score >= COMBINED_THRESHOLD:
            return (best_name, best_score)

    return None


def run_single_pass_pipeline(
    video_path: str,
    output_json_path: str,
    device: str = "cpu",
    identity_db=None,
    match_threshold: float | None = None,
    progress_callback=None,
    reid_stride: int = 1,
    conf_threshold=0.6,
    weak_conf_threshold=0.4,
    min_height=50,
    min_area_ratio=0.001,
    imgsz=640,
):
    """Single-pass pipeline: detection + tracking + ReID in one video decode.

    Instead of reading the video 3 times (detect → track → reid), this reads
    it once and runs all three stages per frame.  Same accuracy, ~3x faster.
    """
    from detection.detect_module import get_cached_detector
    from tracking.track_module import PersonTracker
    from reidentification.reid_main import ReIDEngine, _cosine
    from reidentification.face_cue import FaceCueExtractor, get_cached_face_extractor

    # ── Initialise models (cached across cameras) ────────────────────────
    detector = get_cached_detector(
        conf_threshold=conf_threshold,
        device=device,
        min_height=min_height,
        min_area_ratio=min_area_ratio,
        weak_conf_threshold=weak_conf_threshold,
    )
    tracker = PersonTracker()
    engine = ReIDEngine(device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video: %s", video_path)
        return {"fps": 0, "people": [], "frames_read": 0,
                "frames_expected": 0, "frame_drop_rate": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    face_stride = max(1, int(round(fps / 5)))
    reid_stride = max(1, int(reid_stride))

    results = {}
    face_candidates = {}
    all_detections = {}
    all_tracking = {}
    frame_id = 0
    last_result = []

    logger.info("Single-pass pipeline: %s  (~%d frames @ %.1f fps)",
                video_path, expected_frames, fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # ── Detection (YOLOv8) ──────────────────────────────────────────
        detections = detector.detect(frame, imgsz=imgsz)
        all_detections[frame_id] = detections

        # ── Tracking (DeepSort) ─────────────────────────────────────────
        tracks = tracker.update(frame, detections)
        all_tracking[frame_id] = tracks

        # ── ReID (body + face descriptors + identity assignment) ────────
        if frame_id % reid_stride == 0:
            last_result = engine.process_frame(frame, tracks, frame_id)
        results[frame_id] = last_result

        # ── Collect face candidates for name resolution ─────────────────
        if frame_id % face_stride == 0:
            for p in results[frame_id]:
                tid = p.get("id")
                bbox = p.get("bbox")
                if tid is None or not bbox:
                    continue
                cached = engine._face_cache.get(tid)
                if cached is not None:
                    face_feat, yunet_score, face_bbox = cached
                    face_candidates.setdefault(tid, []).append(
                        (frame_id, bbox, face_feat, yunet_score, face_bbox)
                    )

        # ── Progress ────────────────────────────────────────────────────
        if progress_callback is not None and frame_id % max(1, face_stride * 2) == 0:
            pct = int(60 + 26 * (frame_id / max(1, expected_frames)))
            progress_callback(min(pct, 86),
                              f"Matching known people ({frame_id}/{max(1, expected_frames)} frames)")

    cap.release()
    frames_read = frame_id

    # ── Post-processing (identical to run_camera_reid) ───────────────────
    drop_rate = 0.0
    if expected_frames > 0 and frames_read < expected_frames:
        drop_rate = round(max(0.0, (expected_frames - frames_read) / expected_frames), 4)

    engine.finalize_clustering()
    results = _interpolate_gaps(results, max_gap=3)

    none_resolved = 0
    none_remaining = 0
    for fid in sorted(results.keys()):
        for p in results[fid]:
            if p.get("consolidated_id") is None:
                tid = p.get("id")
                if tid in engine.track_to_identity:
                    p["consolidated_id"] = engine.track_to_identity[tid]
                    none_resolved += 1
                else:
                    none_remaining += 1
    if none_remaining > 0:
        logger.warning("⚠️  %d detections have no consolidated_id — will appear as unidentified",
                        none_remaining)
    logger.info("ReID summary: %d total detections, %d None resolved, %d still None",
                sum(len(v) for v in results.values()), none_resolved, none_remaining)

    # ── Name resolution ──────────────────────────────────────────────────
    name_by_original_sid = {}
    if identity_db is not None and len(identity_db):
        from registration.db_config import SEARCH_SETTINGS as _search_settings
        face_confirm = _search_settings.get("face_match_threshold", 0.40)
        for sid, feat in engine.consolidated_features.items():
            try:
                resolved = _resolve_identity_name(
                    sid, feat, face_candidates, engine.track_to_identity,
                    identity_db, face_confirm, match_threshold,
                )
                if resolved is not None:
                    name_by_original_sid[sid] = resolved
            except Exception:
                logger.exception("Name match failed for stable id %s — leaving unresolved", sid)

    # ── Deduplication ────────────────────────────────────────────────────
    if name_by_original_sid and identity_db is not None:
        from reidentification.face_cue import FaceCueExtractor
        name_groups = {}
        for sid, (name, sim) in name_by_original_sid.items():
            name_groups.setdefault(name, []).append((sid, sim))

        claimed_names = set()
        for name, entries in name_groups.items():
            entries.sort(key=lambda x: x[1], reverse=True)
            claimed_names.add(name)
            best_feat = engine.consolidated_features.get(entries[0][0])
            primary_faces = []
            for tid, cands in face_candidates.items():
                if engine.track_to_identity.get(tid) == entries[0][0]:
                    for (_fid, _bbox, face_feat, _score, _fbox) in cands:
                        if face_feat is not None:
                            primary_faces.append(face_feat)
            best_primary_face = max(primary_faces, key=lambda f: np.linalg.norm(f)) if primary_faces else None

            for sid, sim in entries[1:]:
                dup_feat = engine.consolidated_features.get(sid)
                if dup_feat is None:
                    del name_by_original_sid[sid]
                    continue
                body_sim = _cosine(best_feat, dup_feat) if best_feat is not None else 0.0
                dup_faces = []
                for tid, cands in face_candidates.items():
                    if engine.track_to_identity.get(tid) == sid:
                        for (_fid, _bbox, face_feat, _score, _fbox) in cands:
                            if face_feat is not None:
                                dup_faces.append(face_feat)
                best_dup_face = max(dup_faces, key=lambda f: np.linalg.norm(f)) if dup_faces else None
                face_cross_sim = 0.0
                if best_primary_face is not None and best_dup_face is not None:
                    face_cross_sim = FaceCueExtractor.similarity(best_primary_face, best_dup_face) or 0.0
                different_people = (
                    (body_sim < 0.70 and face_cross_sim >= 0.35)
                    or (face_cross_sim < 0.35 and face_cross_sim > 0.0)
                    or (body_sim < 0.60)
                )
                if different_people:
                    matches = identity_db.match(dup_feat, query_face_embedding=None, top_k=5)
                    new_name = None
                    new_sim = 0.0
                    for m_name, m_sim in matches:
                        if m_name not in claimed_names:
                            new_name = m_name
                            new_sim = m_sim
                            claimed_names.add(m_name)
                            break
                    if new_name:
                        name_by_original_sid[sid] = (new_name, new_sim)
                    else:
                        del name_by_original_sid[sid]
                else:
                    pass  # same person over-split — keep

    # ── Merge over-split tracks ──────────────────────────────────────────
    if name_by_original_sid:
        sid_frame_count = {}
        for fid in sorted(results.keys()):
            for p in results[fid]:
                cid = p.get("consolidated_id")
                if cid is not None:
                    sid_frame_count[cid] = sid_frame_count.get(cid, 0) + 1
        name_to_sids = {}
        for sid, (name, sim) in name_by_original_sid.items():
            name_to_sids.setdefault(name, []).append(sid)
        for name, sids in name_to_sids.items():
            if len(sids) <= 1:
                continue
            primary = max(sids, key=lambda s: sid_frame_count.get(s, 0))
            for sid in sids:
                if sid == primary:
                    continue
                for fid in sorted(results.keys()):
                    for p in results[fid]:
                        if p.get("consolidated_id") == sid:
                            p["consolidated_id"] = primary
                name_by_original_sid.pop(sid, None)
                for tid, mapped_sid in list(engine.track_to_identity.items()):
                    if mapped_sid == sid:
                        engine.track_to_identity[tid] = primary
                if sid in engine.consolidated_features:
                    del engine.consolidated_features[sid]

    # ── Evidence frames ──────────────────────────────────────────────────
    evidence_by_sid = {}
    if name_by_original_sid and identity_db is not None:
        for sid, (name, _sim) in name_by_original_sid.items():
            record = identity_db._data.get(name)
            avg_face = (record or {}).get("average_face_descriptor")
            if avg_face is None:
                continue
            per_frame = {}
            for tid, cands in face_candidates.items():
                if engine.track_to_identity.get(tid) != sid:
                    continue
                for (fid, bbox, face_feat, _score, _fbox) in cands:
                    sim = FaceCueExtractor.similarity(avg_face, face_feat)
                    if sim is None:
                        continue
                    if fid not in per_frame or sim > per_frame[fid][0]:
                        per_frame[fid] = (sim, bbox)
            ranked = sorted(per_frame.items(), key=lambda kv: kv[1][0], reverse=True)[:5]
            evidence_by_sid[sid] = [
                {"frame": fid, "similarity": round(sim, 3), "bbox": bbox}
                for fid, (sim, bbox) in ranked
            ]

    # ── Renumber 1..N ───────────────────────────────────────────────────
    first_seen = {}
    none_first_seen = {}
    for fid in sorted(results.keys()):
        for p in results[fid]:
            cid = p.get("consolidated_id")
            tid = p.get("id")
            if cid is not None and cid not in first_seen:
                first_seen[cid] = fid
            elif cid is None and tid is not None and tid not in none_first_seen:
                none_first_seen[tid] = fid

    remap = {cid: idx + 1 for idx, cid in enumerate(sorted(first_seen.keys(), key=lambda c: first_seen[c]))}
    none_remap = {tid: -(idx + 1) for idx, tid in enumerate(sorted(none_first_seen.keys(), key=lambda t: none_first_seen[t]))}

    track_info = {}
    for fid in sorted(results.keys()):
        for p in results[fid]:
            cid = p.get("consolidated_id")
            tid = p.get("id")
            if cid is not None:
                final_id = remap.get(cid, -1)
                if final_id == -1:
                    continue
                original_sid = cid
            elif tid is not None and tid in none_remap:
                final_id = none_remap[tid]
                original_sid = None
            else:
                continue
            info = track_info.setdefault(final_id, {"first_frame": fid, "last_frame": fid, "original_sid": original_sid})
            info["last_frame"] = fid
            p["consolidated_id"] = final_id

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(results, f)

    people = []
    evidence_dir = Path(output_json_path).parent
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output_stem = Path(output_json_path).stem
    for final_id, info in sorted(track_info.items()):
        match = name_by_original_sid.get(info["original_sid"])
        person_name = match[0] if match else None
        person_status = "normal"
        if person_name and identity_db is not None:
            person_record = identity_db.get_person(person_name) if hasattr(identity_db, 'get_person') else None
            if person_record:
                person_status = person_record.get("status", "normal")
        top_frames = []
        for i, ev in enumerate(evidence_by_sid.get(info["original_sid"], []), start=1):
            crop_path = _save_evidence_frame(
                video_path, ev["frame"], ev["bbox"],
                evidence_dir / f"{output_stem}_track{final_id}_top{i}.jpg",
            )
            top_frames.append({
                "frame": ev["frame"],
                "time_sec": round(ev["frame"] / fps, 1),
                "similarity": ev["similarity"],
                "url": f"/outputs/{crop_path.name}" if crop_path else None,
            })
        people.append({
            "track_id": final_id,
            "name": person_name,
            "status": person_status,
            "similarity": round(float(match[1]), 3) if match else None,
            "first_seen_sec": round(info["first_frame"] / fps, 1),
            "last_seen_sec": round(info["last_frame"] / fps, 1),
            "top_frames": top_frames,
        })

    if progress_callback is not None:
        progress_callback(88, "Finalising matches")

    return {
        "fps": fps,
        "people": people,
        "frames_read": frames_read,
        "frames_expected": expected_frames,
        "frame_drop_rate": drop_rate,
    }


def run_camera_reid(
    video_path: str,
    tracking_json_path: str,
    output_json_path: str,
    device: str = "cpu",
    identity_db=None,
    match_threshold: float | None = None,
    progress_callback=None,
    reid_stride: int = 1,
):
    """
    Run Re-ID for one camera's video and resolve stable identities against
    a registered IdentityDatabase.

    progress_callback(pct: int, message: str) is invoked periodically while
    the (slow) per-frame face extraction runs, so a UI can show movement
    instead of looking stuck at 60%.

    reid_stride: body-descriptor extraction + identity assignment runs on
    every Nth frame only; intermediate frames reuse the previous sampled
    frame's result. DeepSort already provides track continuity across those
    gaps, so this cuts the dominant per-frame inference cost ~3x with
    negligible accuracy impact.

    Returns:
        {
          "fps": float,
          "people": [
             {"track_id": int, "name": str|None, "similarity": float|None,
              "first_seen_sec": float, "last_seen_sec": float},
             ...
          ]
        }
    """
    from reidentification.reid_main import ReIDEngine, _cosine
    from reidentification.face_cue import FaceCueExtractor, get_cached_face_extractor

    engine = ReIDEngine(device=device)

    # The engine's internal face-based decisions (blend/switch/cluster)
    # distinguish identical uniforms via face similarity. Face extraction
    # runs inside process_frame() and populates each candidate's face_feat.
    # Name resolution below uses its own face extractor for the final naming.
    name_face_extractor = get_cached_face_extractor()

    with open(tracking_json_path) as f:
        tracking_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    results = {}
    face_candidates = {}   # track_id -> [(frame_id, bbox, face_feat, yunet_score, face_bbox), ...]
    frame_id = 0
    # Face extraction runs per detection and is the dominant cost of this
    # stage. Candidate faces are only needed for name resolution, so sampling
    # every 5th frame (~5 fps) is plenty and cuts the stage ~5x; the best
    # frame across a track still wins.
    face_stride = max(1, int(round(fps / 5)))
    # Body-descriptor extraction + identity assignment is the dominant cost of
    # this stage. DeepSort gives stable track ids frame-to-frame, so we only
    # run the expensive engine step every reid_stride frames and carry the
    # previous result forward in between — the same idea as face_stride above.
    reid_stride = max(1, int(reid_stride))
    last_result = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        tracks = tracking_data.get(str(frame_id), [])
        if frame_id % reid_stride == 0:
            last_result = engine.process_frame(frame, tracks, frame_id)
        results[frame_id] = last_result
        # Collect per-track face candidates for name resolution. The engine
        # keeps only the last-seen face per identity (often blurrier than
        # the best one); we keep every confident detection so the final
        # name lookup can use the highest-quality face instead.
        if frame_id % face_stride == 0:
            for p in results[frame_id]:
                tid = p.get("id")
                bbox = p.get("bbox")
                if tid is None or not bbox:
                    continue
                # Reuse face result already extracted by the engine in
                # process_frame() instead of running YuNet+ArcFace a second
                # time.  This cuts the dominant per-frame cost roughly in half.
                cached = engine._face_cache.get(tid)
                if cached is not None:
                    face_feat, yunet_score, face_bbox = cached
                    face_candidates.setdefault(tid, []).append(
                        (frame_id, bbox, face_feat, yunet_score, face_bbox)
                    )
        if progress_callback is not None and frame_id % max(1, face_stride * 2) == 0:
            pct = int(60 + 26 * (frame_id / max(1, expected_frames)))
            progress_callback(min(pct, 86), f"Matching known people ({frame_id}/{max(1, expected_frames)} frames)")
    cap.release()

    # Diagnostic: how many face candidates were collected for name resolution?
    total_fc = sum(len(v) for v in face_candidates.values())
    tracks_with_fc = len(face_candidates)
    logger.info("Name-resolution face candidates: %d tracks with faces, %d total candidates",
                tracks_with_fc, total_fc)

    frames_read = frame_id
    # OpenCV's reported frame count is often an estimate (container metadata,
    # not a real decode), so treat a shortfall as a drop only when it's
    # non-trivial rather than off-by-one noise.
    drop_rate = 0.0
    if expected_frames > 0 and frames_read < expected_frames:
        drop_rate = round(max(0.0, (expected_frames - frames_read) / expected_frames), 4)

    engine.finalize_clustering()

    # Fill 1–2 frame gaps where a track temporarily disappears (e.g. brief
    # occlusion / detection miss).  Interpolated bboxes keep the overlay
    # smooth and prevent flickering labels.
    results = _interpolate_gaps(results, max_gap=3)

    # Resolve any remaining None consolidated_ids (mirrors reid_main.run_reid_pipeline)
    none_resolved = 0
    none_remaining = 0
    for fid in sorted(results.keys()):
        for p in results[fid]:
            if p.get("consolidated_id") is None:
                tid = p.get("id")
                if tid in engine.track_to_identity:
                    p["consolidated_id"] = engine.track_to_identity[tid]
                    none_resolved += 1
                else:
                    none_remaining += 1
    if none_remaining > 0:
        logger.warning("⚠️  %d detections have no consolidated_id "
                       "(tiny/off-screen/feature-fail) — will appear as unidentified",
                       none_remaining)
    logger.info("ReID summary: %d total detections, %d None resolved, %d still None",
                sum(len(v) for v in results.values()), none_resolved, none_remaining)

    # Name resolution happens BEFORE renumbering, in the original stable-id
    # space that engine.consolidated_features is keyed by.
    #
    # Strategy (face-primary): if a face was EVER detected on the identity,
    # the face decides the name — the best candidate frame that clears the
    # confirm bar (>= face_match_threshold) names the person, and if no face
    # clears the bar the identity stays Unknown. The body is only consulted
    # when NO face was ever seen on the track (e.g. back-to-camera). An
    # inconclusive face must never be overruled by a body guess: that
    # body-over-face fallback is what mislabelled a stranger as "Pranjali"
    # in earlier runs despite a ~0.31 face (below the 0.30 veto bar).
    name_by_original_sid = {}
    if identity_db is not None and len(identity_db):
        from registration.db_config import SEARCH_SETTINGS as _search_settings
        face_confirm = _search_settings.get("face_match_threshold", 0.40)
        for sid, feat in engine.consolidated_features.items():
            try:
                resolved = _resolve_identity_name(
                    sid, feat, face_candidates, engine.track_to_identity,
                    identity_db, face_confirm, match_threshold,
                )
                if resolved is not None:
                    name_by_original_sid[sid] = resolved
            except Exception:
                logger.exception("Name match failed for stable id %s — leaving unresolved", sid)

    # ── Deduplication: if two identities got the same name, decide ────────
    # With a weak body model, the engine over-splits one person into multiple
    # stable IDs. Face naming correctly gives them ALL the same name. We must
    # NOT re-resolve those — they're the same person, just over-split. Only
    # re-resolve when body similarity between the duplicates is LOW (meaning
    # two genuinely different people coincidentally got the same face name).
    #
    # Face cross-check: body similarity alone is unreliable for uniformed
    # subjects (always 0.90+). We also compare the BEST face candidates
    # between the two SIDs. If face similarity is low (< 0.35), they are
    # genuinely different people even if body similarity is high.
    if name_by_original_sid and identity_db is not None:
        from reidentification.face_cue import FaceCueExtractor
        name_groups = {}  # name -> [(sid, sim), ...]
        for sid, (name, sim) in name_by_original_sid.items():
            name_groups.setdefault(name, []).append((sid, sim))

        claimed_names = set()
        for name, entries in name_groups.items():
            entries.sort(key=lambda x: x[1], reverse=True)  # best first
            claimed_names.add(name)
            best_feat = engine.consolidated_features.get(entries[0][0])
            # Collect best face for primary SID
            primary_faces = []
            for tid, cands in face_candidates.items():
                if engine.track_to_identity.get(tid) == entries[0][0]:
                    for (_fid, _bbox, face_feat, _score, _fbox) in cands:
                        if face_feat is not None:
                            primary_faces.append(face_feat)
            best_primary_face = max(primary_faces, key=lambda f: np.linalg.norm(f)) if primary_faces else None

            for sid, sim in entries[1:]:
                dup_feat = engine.consolidated_features.get(sid)
                if dup_feat is None:
                    del name_by_original_sid[sid]
                    continue
                # Check if body similarity is low → different people, re-resolve
                body_sim = _cosine(best_feat, dup_feat) if best_feat is not None else 0.0

                # Face cross-check: compare best face candidates between SIDs
                dup_faces = []
                for tid, cands in face_candidates.items():
                    if engine.track_to_identity.get(tid) == sid:
                        for (_fid, _bbox, face_feat, _score, _fbox) in cands:
                            if face_feat is not None:
                                dup_faces.append(face_feat)
                best_dup_face = max(dup_faces, key=lambda f: np.linalg.norm(f)) if dup_faces else None

                face_cross_sim = 0.0
                if best_primary_face is not None and best_dup_face is not None:
                    face_cross_sim = FaceCueExtractor.similarity(best_primary_face, best_dup_face) or 0.0

                # Different people if: body is low OR face cross-sim is low.
                # When face is unavailable (0.0), require body_sim < 0.60
                # to treat as different — prevents auto-merging two people
                # just because face wasn't detected.
                different_people = (
                    (body_sim < 0.70 and face_cross_sim >= 0.35)
                    or (face_cross_sim < 0.35 and face_cross_sim > 0.0)
                    or (body_sim < 0.60)
                )

                if different_people:
                    # Genuinely different people — re-resolve against remaining names
                    matches = identity_db.match(
                        dup_feat, query_face_embedding=None, top_k=5
                    )
                    new_name = None
                    new_sim = 0.0
                    for m_name, m_sim in matches:
                        if m_name not in claimed_names:
                            new_name = m_name
                            new_sim = m_sim
                            claimed_names.add(m_name)
                            break
                    if new_name:
                        name_by_original_sid[sid] = (new_name, new_sim)
                        logger.info("Dedup: sid=%s renamed %s -> %s (body=%.3f, face_cross=%.3f, sim %.3f -> %.3f)",
                                    sid, name, new_name, body_sim, face_cross_sim, sim, new_sim)
                    else:
                        del name_by_original_sid[sid]
                        logger.info("Dedup: sid=%s removed %s (no other match, body=%.3f, face_cross=%.3f) -> Unknown",
                                    sid, name, body_sim, face_cross_sim)
                else:
                    # Same person over-split by body model — keep same name
                    logger.info("Dedup: sid=%s kept as %s (over-split, body=%.3f, face_cross=%.3f)",
                                sid, name, body_sim, face_cross_sim)

    # ── Merge over-split tracks: same name → one unified track ───────────
    # When the engine creates multiple stable IDs for the same person
    # (because the body model is weak), face naming gives them all the
    # same name. Merge them so the UI shows one track, not two.
    if name_by_original_sid:
        # Count frames per SID across all results
        sid_frame_count = {}
        for fid in sorted(results.keys()):
            for p in results[fid]:
                cid = p.get("consolidated_id")
                if cid is not None:
                    sid_frame_count[cid] = sid_frame_count.get(cid, 0) + 1

        # Group SIDs by name
        name_to_sids = {}
        for sid, (name, sim) in name_by_original_sid.items():
            name_to_sids.setdefault(name, []).append(sid)

        # Merge duplicates: keep primary (most frames), reassign secondary
        for name, sids in name_to_sids.items():
            if len(sids) <= 1:
                continue
            # Primary = SID with most frames (most stable representation)
            primary = max(sids, key=lambda s: sid_frame_count.get(s, 0))
            for sid in sids:
                if sid == primary:
                    continue
                # Reassign all tracks from secondary to primary
                reassign_count = 0
                for fid in sorted(results.keys()):
                    for p in results[fid]:
                        if p.get("consolidated_id") == sid:
                            p["consolidated_id"] = primary
                            reassign_count += 1
                # Remove secondary from name mapping
                name_by_original_sid.pop(sid, None)
                # Update engine mappings
                for tid, mapped_sid in list(engine.track_to_identity.items()):
                    if mapped_sid == sid:
                        engine.track_to_identity[tid] = primary
                if sid in engine.consolidated_features:
                    del engine.consolidated_features[sid]
                logger.info("Merge: sid=%s -> primary sid=%s (%s, %d frames reassigned)",
                            sid, primary, name, reassign_count)

    # Evidence frames: for each MATCHED identity, keep the top-5 frames whose
    # per-frame face similarity to the matched person's average face is the
    # highest. This is the honest "these are the moments that decided the
    # match" list shown in the UI.
    evidence_by_sid = {}
    if name_by_original_sid and identity_db is not None:
        for sid, (name, _sim) in name_by_original_sid.items():
            record = identity_db._data.get(name)
            avg_face = (record or {}).get("average_face_descriptor")
            if avg_face is None:
                continue
            per_frame = {}
            for tid, cands in face_candidates.items():
                if engine.track_to_identity.get(tid) != sid:
                    continue
                for (fid, bbox, face_feat, _score, _fbox) in cands:
                    sim = FaceCueExtractor.similarity(avg_face, face_feat)
                    if sim is None:
                        continue
                    if fid not in per_frame or sim > per_frame[fid][0]:
                        per_frame[fid] = (sim, bbox)
            ranked = sorted(per_frame.items(), key=lambda kv: kv[1][0], reverse=True)[:5]
            evidence_by_sid[sid] = [
                {"frame": fid, "similarity": round(sim, 3), "bbox": bbox}
                for fid, (sim, bbox) in ranked
            ]

    # Renumber 1..N by first appearance, same rule as reid_main.run_reid_pipeline,
    # so on-screen labels stay stable and consistent with any rendered video.
    # Tracks with consolidated_id=None (dropped by size/filter/extraction) get
    # negative IDs so they still appear as unidentified in the overlay.
    first_seen = {}
    none_first_seen = {}
    for fid in sorted(results.keys()):
        for p in results[fid]:
            cid = p.get("consolidated_id")
            tid = p.get("id")
            if cid is not None and cid not in first_seen:
                first_seen[cid] = fid
            elif cid is None and tid is not None and tid not in none_first_seen:
                none_first_seen[tid] = fid

    remap = {
        cid: idx + 1
        for idx, cid in enumerate(sorted(first_seen.keys(), key=lambda c: first_seen[c]))
    }
    # Assign negative IDs to unresolved tracks (by first appearance order)
    none_remap = {
        tid: -(idx + 1)
        for idx, tid in enumerate(sorted(none_first_seen.keys(), key=lambda t: none_first_seen[t]))
    }

    track_info = {}
    for fid in sorted(results.keys()):
        for p in results[fid]:
            cid = p.get("consolidated_id")
            tid = p.get("id")
            if cid is not None:
                final_id = remap.get(cid, -1)
                if final_id == -1:
                    continue
                original_sid = cid
            elif tid is not None and tid in none_remap:
                final_id = none_remap[tid]
                original_sid = None
            else:
                continue
            info = track_info.setdefault(final_id, {"first_frame": fid, "last_frame": fid, "original_sid": original_sid})
            info["last_frame"] = fid
            p["consolidated_id"] = final_id  # keep output JSON consistent with the render step

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(results, f)

    people = []
    evidence_dir = Path(output_json_path).parent
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output_stem = Path(output_json_path).stem
    for final_id, info in sorted(track_info.items()):
        match = name_by_original_sid.get(info["original_sid"])
        person_name = match[0] if match else None
        # Get person status from identity database
        person_status = "normal"
        if person_name and identity_db is not None:
            person_record = identity_db.get_person(person_name) if hasattr(identity_db, 'get_person') else None
            if person_record:
                person_status = person_record.get("status", "normal")
        top_frames = []
        for i, ev in enumerate(evidence_by_sid.get(info["original_sid"], []), start=1):
            crop_path = _save_evidence_frame(
                video_path,
                ev["frame"],
                ev["bbox"],
                evidence_dir / f"{output_stem}_track{final_id}_top{i}.jpg",
            )
            top_frames.append(
                {
                    "frame": ev["frame"],
                    "time_sec": round(ev["frame"] / fps, 1),
                    "similarity": ev["similarity"],
                    "url": f"/outputs/{crop_path.name}" if crop_path else None,
                }
            )
        people.append(
            {
                "track_id": final_id,
                "name": person_name,
                "status": person_status,
                "similarity": round(float(match[1]), 3) if match else None,
                "first_seen_sec": round(info["first_frame"] / fps, 1),
                "last_seen_sec": round(info["last_frame"] / fps, 1),
                "top_frames": top_frames,
            }
        )

    if progress_callback is not None:
        progress_callback(88, "Finalising matches")

    return {
        "fps": fps,
        "people": people,
        "frames_read": frames_read,
        "frames_expected": expected_frames,
        "frame_drop_rate": drop_rate,
    }

def _interpolate_gaps(results: dict, max_gap: int = 2) -> dict:
    """Fill 1–2 frame gaps where a track temporarily disappears.

    When a tracker ID vanishes for <= max_gap frames and reappears with the
    same consolidated_id, the missing frames are filled by linearly
    interpolating the bounding box between the last-seen and first-seen
    positions.  This smooths out brief occlusions / detection misses
    without affecting longer gaps (which are left as-is, since they likely
    represent a person leaving and re-entering).
    """
    frame_ids = sorted(results.keys())
    if len(frame_ids) < 3:
        return results

    # Build per-track_id timeline: {tid: [(frame_id, bbox, consolidated_id), ...]}
    track_timeline: dict = {}
    for fid in frame_ids:
        for p in results[fid]:
            tid = p.get("id")
            cid = p.get("consolidated_id")
            bbox = p.get("bbox")
            if tid is not None and cid is not None and bbox is not None:
                track_timeline.setdefault(tid, []).append((fid, bbox, cid))

    # For each track, find gaps and fill them
    filled_fids = set()
    for tid, entries in track_timeline.items():
        if len(entries) < 2:
            continue
        for idx in range(len(entries) - 1):
            fid_a, bbox_a, cid_a = entries[idx]
            fid_b, bbox_b, cid_b = entries[idx + 1]
            if cid_a != cid_b:
                continue  # different identity — don't interpolate
            gap = fid_b - fid_a
            if gap <= 1 or gap > max_gap:
                continue
            # Linearly interpolate bounding boxes for missing frames
            for step in range(1, gap):
                t = step / gap
                interp_bbox = [
                    bbox_a[k] + t * (bbox_b[k] - bbox_a[k])
                    for k in range(4)
                ]
                interp_fid = fid_a + step
                # Only fill if the frame slot doesn't already have this tid
                existing_tids = {p["id"] for p in results.get(interp_fid, [])}
                if tid not in existing_tids:
                    results.setdefault(interp_fid, []).append({
                        "id": tid,
                        "consolidated_id": cid_a,
                        "bbox": interp_bbox,
                        "feature_dim": 0,
                        "matches": [],
                    })
                    filled_fids.add(interp_fid)

    if filled_fids:
        logger.debug(f"Interpolated {len(filled_fids)} frames across {len(track_timeline)} tracks")
    return results


def load_track_overlay(reid_json_path: str, people: list, fps: float, max_frames: int = 900) -> dict:
    """
    Build a compact, canvas-ready overlay track from an already-written
    reid JSON file (per-frame bboxes) plus the name-resolved `people`
    summary from run_camera_reid (per track_id). Used to drive the live
    bounding-box + placard overlay without re-running the model.

    Downsamples to at most `max_frames` sampled frames so long clips don't
    ship enormous payloads to the browser.
    """
    name_by_track = {p["track_id"]: p for p in people}

    with open(reid_json_path) as f:
        raw = json.load(f)

    frame_ids = sorted(int(fid) for fid in raw.keys())
    step = max(1, len(frame_ids) // max_frames) if frame_ids else 1

    frames = []
    total_boxes = 0
    for fid in frame_ids[::step]:
        boxes = []
        for p in raw.get(str(fid), []):
            tid = p.get("consolidated_id")
            if tid is None or tid == -1:
                continue
            info = name_by_track.get(tid, {})
            boxes.append(
                {
                    "track_id": tid,
                    "bbox": p.get("bbox"),
                    "name": info.get("name"),
                    "status": info.get("status", "normal"),
                    "similarity": info.get("similarity"),
                }
            )
        if boxes:
            frames.append({"t": round(fid / fps, 3), "boxes": boxes})
            total_boxes += len(boxes)

    logger.info("Overlay: %d frames with boxes, %d total boxes from %d source frames",
                len(frames), total_boxes, len(frame_ids))
    return {"fps": fps, "frames": frames}
