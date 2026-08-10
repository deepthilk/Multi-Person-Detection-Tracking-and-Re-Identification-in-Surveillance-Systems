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

logger = logging.getLogger(__name__)


def _frame_keys(results):
    """Numeric frame keys only — skips reserved "_"-prefixed metadata keys
    (__tracks__, __verify_faces__) that live in the same dict."""
    return (k for k in results if isinstance(k, int))


def run_camera_reid(
    video_path: str,
    tracking_json_path: str,
    output_json_path: str,
    device: str = "cpu",
    identity_db=None,
    match_threshold: float = None,
    sample_every: int = 6,
    max_faces_per_track: int = 30,
    face_upsample: int = 3,
    stride: int = 1,
    verify_stride: int = 6,
):
    """
    Run Re-ID for one camera's video and resolve stable identities against
    a registered IdentityDatabase using face + body-appearance matching
    (registration.identity_db.match_multimodal).

    Args:
        stride: process only every Nth frame — the same person appears across
            many consecutive frames, so identity results barely change while
            runtime drops ~stride× (the main CPU-only speed knob).

    Returns:
        {
          "fps": float,
          "people": [
             {"track_id": int, "name": str|None, "similarity": float|None,
              "face_sim": float|None, "cues": [str], 
              "first_seen_sec": float, "last_seen_sec": float},
             ...
          ]
        }
    """
    from reidentification.insight_face import get_shared_extractor
    from reidentification.reid_main import ReIDEngine

    engine = ReIDEngine(device=device)
    # MUST match the face extractor used at registration time (512-dim ArcFace
    # via InsightFace). The dlib FaceCueExtractor returns 128-dim vectors that
    # fail the dot-product in identity_db.match_multimodal against the 512-dim
    # gallery, silently leaving every identity unresolved ("deeps not found").
    gallery_face_extractor = get_shared_extractor()

    with open(tracking_json_path) as f:
        tracking_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Face gallery per tracker id — every processed frame contributes a face,
    # capped by max_faces_per_track (stride already thins the cost).
    tid_faces = {}

    results = {}
    frame_id = 0
    stride = max(1, int(stride))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if (frame_id - 1) % stride != 0:
            continue
        tracks = tracking_data.get(str(frame_id), [])
        frame_results = engine.process_frame(frame, tracks, frame_id)
        results[frame_id] = frame_results

        for p in frame_results:
            tid = p.get("id")
            if tid is None or p.get("feature_dim", 0) == 0:
                continue
            # Keep a face from EVERY processed frame (stride already thins the
            # cost), capped by max_faces_per_track. Modulo-thinning on top of
            # stride starves borderline matches: a real same-person face at
            # ~0.53 (vs 0.45 threshold) was sampled out and the person came
            # back "not found" even though she was in the video.
            if len(tid_faces.get(tid, [])) >= max_faces_per_track:
                continue
            face = gallery_face_extractor.extract(frame, p["bbox"])
            if face is not None:
                # (frame, embedding) — the frame lets the face-verification
                # pass (web/face_verify.py) run WITHOUT re-decoding the video.
                tid_faces.setdefault(tid, []).append((frame_id, face))
    cap.release()

    frames_read = frame_id
    # OpenCV's reported frame count is often an estimate (container metadata,
    # not a real decode), so treat a shortfall as a drop only when it's
    # non-trivial rather than off-by-one noise.
    drop_rate = 0.0
    if expected_frames > 0 and frames_read < expected_frames:
        drop_rate = round(max(0.0, (expected_frames - frames_read) / expected_frames), 4)

    engine.finalize_clustering()

    # Sync every frame entry's consolidated_id to the FINAL id_mapping.
    # finalize_clustering may have split/merged identities (co-occurrence fix,
    # face-based split of false merges) AFTER the per-frame results were
    # written, so they can be stale — without this, a split identity still
    # renders under its old id (e.g. "id1 shared between two different people").
    # Mirrors the sync reid_main.run_reid_pipeline performs.
    for fid in sorted(_frame_keys(results)):
        for p in results[fid]:
            tid = p.get("id")
            if tid in engine.id_mapping and p.get("consolidated_id") != engine.id_mapping[tid]:
                p["consolidated_id"] = engine.id_mapping[tid]

    # Pool faces per stable identity using the FINAL tracker->sid mapping.
    sid_faces = {}
    for tid, sid in engine.id_mapping.items():
        faces = [f for _, f in tid_faces.get(tid, [])]
        if faces:
            sid_faces.setdefault(sid, []).extend(faces)

    # Per-tracker (frame, embedding) cache for the face-verification pass.
    # Persisted under a reserved "_" key so it survives the round-trip without
    # colliding with numeric frame keys; the pass reads it instead of decoding
    # the video again to re-extract faces.
    results["__verify_faces__"] = {
        str(tid): [[fid, f.tolist()] for fid, f in arr]
        for tid, arr in tid_faces.items()
        if arr
    }

    # Resolve any remaining None consolidated_ids (mirrors reid_main.run_reid_pipeline)
    for fid in sorted(_frame_keys(results)):
        for p in results[fid]:
            if p.get("consolidated_id") is None:
                tid = p.get("id")
                if tid in engine.track_to_identity:
                    p["consolidated_id"] = engine.track_to_identity[tid]

    # Remap orphan sids discovered during finalize_clustering (trackers briefly
    # assigned a new identity, then switched back — mirrors reid_main).
    if engine.orphan_remap:
        for fid in sorted(_frame_keys(results)):
            for p in results[fid]:
                cid = p.get("consolidated_id")
                if cid in engine.orphan_remap:
                    p["consolidated_id"] = engine.orphan_remap[cid]

    # Name resolution happens BEFORE renumbering, in the original stable-id
    # space that engine.consolidated_features is keyed by. Uses face + body
    # appearance fusion; degrades to appearance-only when no faces exist.
    name_by_original_sid = {}
    if identity_db is not None and len(identity_db):
        for sid, feat in engine.consolidated_features.items():
            try:
                matches = identity_db.match_multimodal(
                    feat, sid_faces.get(sid, []),
                    top_k=1, threshold=match_threshold,
                )
            except Exception:
                logger.exception("Name match failed for stable id %s — leaving unresolved", sid)
                matches = []
            if matches:
                m = matches[0]
                name_by_original_sid[sid] = (m["name"], m["score"], m["face_sim"], m["cues"])

    # Renumber 1..N by first appearance, same rule as reid_main.run_reid_pipeline,
    # so on-screen labels stay stable and consistent with any rendered video.
    first_seen = {}
    for fid in sorted(_frame_keys(results)):
        for p in results[fid]:
            cid = p.get("consolidated_id")
            if cid is not None and cid not in first_seen:
                first_seen[cid] = fid

    remap = {
        cid: idx + 1
        for idx, cid in enumerate(sorted(first_seen.keys(), key=lambda c: first_seen[c]))
    }

    track_info = {}
    for fid in sorted(_frame_keys(results)):
        for p in results[fid]:
            cid = p.get("consolidated_id")
            if cid is None:
                continue
            final_id = remap.get(cid, -1)
            if final_id == -1:
                continue
            info = track_info.setdefault(final_id, {"first_frame": fid, "last_frame": fid, "original_sid": cid})
            info["last_frame"] = fid
            p["consolidated_id"] = final_id  # keep output JSON consistent with the render step

    # Per-track evidence, persisted under a reserved "__tracks__" key so the
    # cross-camera pass (web/cross_camera.py) and the manual-correction flow
    # can unify/learn WITHOUT re-running the engine: face gallery (512-dim
    # InsightFace, same space as registration), track-average appearance
    # descriptor, and the auto-resolved name. Numeric frame keys can never
    # collide with "__tracks__".
    tracks_payload = {}
    for final_id, info in sorted(track_info.items()):
        sid = info["original_sid"]
        faces = [f.tolist() for f in sid_faces.get(sid, [])]
        desc = engine.consolidated_features.get(sid)
        match = name_by_original_sid.get(sid)
        tracks_payload[str(final_id)] = {
            "original_sid": sid,
            "faces": faces,
            "mean_feature": desc.tolist() if desc is not None else None,
            "name": match[0] if match else None,
            "similarity": round(float(match[1]), 3) if match else None,
            "face_sim": round(float(match[2]), 3) if match and match[2] is not None else None,
            "cues": match[3] if match else [],
        }
    results["__tracks__"] = tracks_payload

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    people = []
    for final_id, info in sorted(track_info.items()):
        match = name_by_original_sid.get(info["original_sid"])
        people.append(
            {
                "track_id": final_id,
                "name": match[0] if match else None,
                "similarity": round(float(match[1]), 3) if match else None,
                "face_sim": round(float(match[2]), 3) if match and match[2] is not None else None,
                "cues": match[3] if match else [],
                "first_seen_sec": round(info["first_frame"] / fps, 1),
                "last_seen_sec": round(info["last_frame"] / fps, 1),
            }
        )

    # Face-first verification pass: the engine's per-frame ids and names come
    # from a body-appearance + face blend that is unreliable when people wear
    # IDENTICAL uniforms (appearance scores are ~equal for everyone, so ids
    # and names become coin-flips). The pass re-asserts identity from FACE
    # evidence only (web/face_verify.py): corrects tracker swaps, splits
    # appearance-based false merges and wrong names, rewrites the reid JSON in
    # place, and returns a corrected people summary. It is conservative and
    # leaves the engine output untouched when no decisive face evidence exists
    # (or when no registration DB is available).
    corrected = None
    if identity_db is not None:
        from web.face_verify import verify_and_fix
        corrected = verify_and_fix(
            video_path, tracking_json_path, output_json_path, identity_db,
            stride=verify_stride, fps=fps, max_faces=max_faces_per_track,
        )
    if corrected is not None:
        people = corrected

    return {
        "fps": fps,
        "people": people,
        "frames_read": frames_read,
        "frames_expected": expected_frames,
        "frame_drop_rate": drop_rate,
    }


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

    # Fill per-track gaps (the reid step samples frames at stride > 1) so the
    # live overlay shows a continuously moving box instead of blinking on and
    # off between sampled frames.
    from utils import fill_track_gaps
    raw = fill_track_gaps(raw)

    frame_ids = sorted(int(fid) for fid in raw if fid.lstrip('-').isdigit())
    step = max(1, len(frame_ids) // max_frames) if frame_ids else 1

    frames = []
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
                    "similarity": info.get("similarity"),
                }
            )
        if boxes:
            frames.append({"t": round(fid / fps, 3), "boxes": boxes})

    return {"fps": fps, "frames": frames}
