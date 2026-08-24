"""
Baseline per-stage profiler for the web per-camera Re-ID flow.
==============================================================

Mirrors the exact frame loop of web/multicam_pipeline.run_camera_reid() and
times each stage independently so the PERFORMANCE_REPORT.md estimates can be
checked against real numbers before any speed-up is applied:

    * ReIDEngine init        — ResNet-50 + Re-ID head load (includes the
                               engine's own InsightFace model load)
    * shared InsightFace     — get_shared_extractor() load (the SECOND
                               InsightFace model set per camera job)
    * decode                 — cv2.VideoCapture.read()
    * engine.process_frame   — per-frame Re-ID: 698-dim descriptor extraction
                               (ResNet + zonal HSV + LBP + proportions) per
                               person, internal face cue, and identity matching
    * gallery face pass      — the web flow's second per-person InsightFace
                               extraction (get_shared_extractor)
    * finalize_clustering    — cross-frame identity consolidation
    * JSON serialization     — indent=2 vs compact output size

Usage:
    python profile_pipeline.py                                   # cam1, cpu, 120 frames
    python profile_pipeline.py --device cuda --stride 6 --max-frames 400
    python profile_pipeline.py --video input/video2.mp4 --tracking outputs/multicam/tracking/cam2_tracking.json
"""

import argparse
import json
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Baseline per-stage profiler for the per-camera Re-ID flow")
    parser.add_argument("--video", type=str, default="input/video1.mp4")
    parser.add_argument("--tracking", type=str, default="outputs/multicam/tracking/cam1_tracking.json")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--skip-face", action="store_true",
                        help="Skip the gallery face pass to isolate pure Re-ID cost")
    parser.add_argument("--output", type=str, default="outputs/profiling/baseline.json")
    args = parser.parse_args()

    from reidentification.insight_face import get_shared_extractor
    from reidentification.reid_main import ReIDEngine

    t0 = time.perf_counter()
    engine = ReIDEngine(device=args.device)
    t_engine_init = time.perf_counter() - t0

    t0 = time.perf_counter()
    face_ex = get_shared_extractor()
    t_face_load = time.perf_counter() - t0

    tracking = json.loads(Path(args.tracking).read_text(encoding="utf-8"))

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

    stride = max(1, int(args.stride))
    max_frames = int(args.max_frames)
    max_faces_per_track = 30

    t_decode = 0.0
    t_engine = 0.0
    t_face = 0.0
    n_frames_read = 0
    n_frames_processed = 0
    n_persons = 0
    n_face_attempts = 0
    n_face_hits = 0
    tid_faces = {}

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t_decode += time.perf_counter() - t0
        if not ret:
            break
        n_frames_read += 1
        if (n_frames_read - 1) % stride != 0:
            continue
        n_frames_processed += 1

        tracks = tracking.get(str(n_frames_read), [])

        t0 = time.perf_counter()
        frame_results = engine.process_frame(frame, tracks, n_frames_read)
        t_engine += time.perf_counter() - t0

        for person in frame_results:
            tid = person.get("id")
            if tid is None or person.get("feature_dim", 0) == 0:
                continue
            n_persons += 1
            if len(tid_faces.get(tid, [])) >= max_faces_per_track:
                continue
            if not args.skip_face:
                n_face_attempts += 1
                t0 = time.perf_counter()
                face = face_ex.extract(frame, person["bbox"])
                t_face += time.perf_counter() - t0
                if face is not None:
                    n_face_hits += 1
                    tid_faces.setdefault(tid, []).append((n_frames_read, face))

        if n_frames_processed >= max_frames:
            break

    cap.release()

    t0 = time.perf_counter()
    engine.finalize_clustering()
    t_finalize = time.perf_counter() - t0

    payload = {
        "__verify_faces__": {
            str(tid): [[fid, f.tolist()] for fid, f in arr]
            for tid, arr in tid_faces.items()
            if arr
        },
    }

    t0 = time.perf_counter()
    indent_json = json.dumps(payload, indent=2)
    t_indent = time.perf_counter() - t0
    t0 = time.perf_counter()
    compact_json = json.dumps(payload, separators=(",", ":"))
    t_compact = time.perf_counter() - t0

    frames_proc = max(1, n_frames_processed)
    total_loop = t_engine + t_face + t_decode
    per_frame = total_loop / frames_proc
    engine_fps = 1.0 / per_frame if per_frame > 0 else 0.0
    projected_full = (frames_meta / stride) * per_frame if frames_meta else None

    report = {
        "video": args.video,
        "tracking": args.tracking,
        "device": str(engine.device),
        "frame_w": frame_w,
        "frame_h": frame_h,
        "fps": round(fps, 2),
        "frames_meta": frames_meta,
        "frames_read": n_frames_read,
        "frames_processed": n_frames_processed,
        "persons_seen": n_persons,
        "face_attempts": n_face_attempts,
        "face_hits": n_face_hits,
        "stride": stride,
        "loads_seconds": {
            "engine_init": round(t_engine_init, 2),
            "shared_insightface": round(t_face_load, 2),
        },
        "totals_seconds": {
            "decode": round(t_decode, 3),
            "engine_process_frame": round(t_engine, 3),
            "gallery_face": round(t_face, 3),
            "finalize_clustering": round(t_finalize, 3),
            "frame_loop": round(total_loop, 3),
        },
        "per_frame_ms": {
            "decode": round(1000 * t_decode / frames_proc, 2),
            "engine_process_frame": round(1000 * t_engine / frames_proc, 2),
            "gallery_face": round(1000 * t_face / frames_proc, 2),
            "total": round(1000 * per_frame, 2),
        },
        "per_person_ms": {
            "engine_process_frame": round(1000 * t_engine / max(1, n_persons), 2),
            "gallery_face": round(1000 * t_face / max(1, n_face_attempts), 2),
        },
        "achievable_fps": round(engine_fps, 2),
        "projected_full_clip_seconds": round(projected_full, 1) if projected_full else None,
        "json_bytes": {
            "indent_2": len(indent_json.encode("utf-8")),
            "compact": len(compact_json.encode("utf-8")),
            "savings_pct": round(100 * (1 - len(compact_json) / max(1, len(indent_json))), 2),
        },
        "json_serialize_ms": {
            "indent_2": round(1000 * t_indent, 2),
            "compact": round(1000 * t_compact, 2),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"wrote baseline to {out}")


if __name__ == "__main__":
    main()
