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
):
    """
    Run Re-ID for one camera's video and resolve stable identities against
    a registered IdentityDatabase using face + body-appearance matching
    (registration.identity_db.match_multimodal).

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
    from reidentification.face_cue import FaceCueExtractor
    from reidentification.reid_main import ReIDEngine

    engine = ReIDEngine(device=device)
    gallery_face_extractor = FaceCueExtractor(upsample_times=face_upsample)

    with open(tracking_json_path) as f:
        tracking_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Face gallery per tracker id, sampled to keep overhead low (the engine
    # already runs face detection internally, so we only add a fraction more).
    tid_faces = {}
    tid_face_count = {}

    results = {}
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        tracks = tracking_data.get(str(frame_id), [])
        frame_results = engine.process_frame(frame, tracks, frame_id)
        results[frame_id] = frame_results

        for p in frame_results:
            tid = p.get("id")
            if tid is None or p.get("feature_dim", 0) == 0:
                continue
            tid_face_count.setdefault(tid, 0)
            if len(tid_faces.get(tid, [])) >= max_faces_per_track:
                continue
            if tid_face_count[tid] % sample_every != 0:
                tid_face_count[tid] += 1
                continue
            tid_face_count[tid] += 1
            face = gallery_face_extractor.extract(frame, p["bbox"])
            if face is not None:
                tid_faces.setdefault(tid, []).append(face)
    cap.release()

    frames_read = frame_id
    # OpenCV's reported frame count is often an estimate (container metadata,
    # not a real decode), so treat a shortfall as a drop only when it's
    # non-trivial rather than off-by-one noise.
    drop_rate = 0.0
    if expected_frames > 0 and frames_read < expected_frames:
        drop_rate = round(max(0.0, (expected_frames - frames_read) / expected_frames), 4)

    engine.finalize_clustering()

    # Pool faces per stable identity using the FINAL tracker->sid mapping.
    sid_faces = {}
    for tid, sid in engine.id_mapping.items():
        faces = tid_faces.get(tid, [])
        if faces:
            sid_faces.setdefault(sid, []).extend(faces)

    # Resolve any remaining None consolidated_ids (mirrors reid_main.run_reid_pipeline)
    for fid in sorted(results.keys()):
        for p in results[fid]:
            if p.get("consolidated_id") is None:
                tid = p.get("id")
                if tid in engine.track_to_identity:
                    p["consolidated_id"] = engine.track_to_identity[tid]

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
    for fid in sorted(results.keys()):
        for p in results[fid]:
            cid = p.get("consolidated_id")
            if cid is not None and cid not in first_seen:
                first_seen[cid] = fid

    remap = {
        cid: idx + 1
        for idx, cid in enumerate(sorted(first_seen.keys(), key=lambda c: first_seen[c]))
    }

    track_info = {}
    for fid in sorted(results.keys()):
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

    frame_ids = sorted(int(fid) for fid in raw.keys())
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
