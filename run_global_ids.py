"""
Assign GLOBAL identity IDs across videos, then render consistent-ID videos.
============================================================================

Each video's `consolidated_id` is local — the same physical person gets a
different number in different videos. This tool matches identities ACROSS
videos (same appearance -> same global_id) and re-renders each video with
the global IDs, so "the girl" is labelled the same in every clip.

Flow:
  1. Load each video's existing outputs/runs/<name>/reid_results.json.
  2. Re-extract a per-identity 698-dim appearance descriptor (sampled frames)
     using the SAME Re-ID backbone the pipeline uses.
  3. Feed all videos into reidentification.cross_camera_match.CrossCameraMatcher
     -> one global_id per physical person.
  4. Write outputs/global/global_identities.json (combined structure, same
     schema run_integrated_pipeline.py produces).
  5. Render each video -> outputs/runs/<name>/reid_global.mp4 labelled with
     the global IDs.

Usage:
    python run_global_ids.py --videos "input1,input 2"
"""
import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _identity_descriptors(vname, engine, sample=15):
    """Re-extract one averaged 698-dim descriptor per stable identity from a
    video's already-computed reid_results.json (sampled frames)."""
    results = json.load(open(f"outputs/runs/{vname}/reid_results.json", encoding="utf-8"))
    cap = cv2.VideoCapture(f"input/{vname}.mp4")
    by_sid = {}
    for fid_str, people in results.items():
        fid = int(fid_str)
        for p in people:
            sid = p.get("consolidated_id")
            if sid is None or sid == -1 or p.get("feature_dim", 0) == 0:
                continue
            by_sid.setdefault(sid, []).append((fid, p["bbox"]))

    descriptors = {}
    for sid, frames in by_sid.items():
        frames = sorted(frames)
        step = max(1, len(frames) // sample)
        feats = []
        for fid, bbox in frames[::step]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
            ok, frame = cap.read()
            if not ok:
                continue
            feat = engine.extract_feature(frame, bbox)
            if feat is not None:
                feats.append(feat)
        if feats:
            v = np.mean(feats, 0)
            descriptors[sid] = v / (np.linalg.norm(v) + 1e-8)
    cap.release()
    return descriptors


def _identity_face_descriptors(vname, sample=25):
    """One 128-dim face embedding per identity via insightface (dress-invariant
    side channel for cross-video matching). Returns {} if insightface is not
    installed or no faces can be found."""
    import cv2
    import numpy as np

    try:
        from insightface.app import FaceAnalysis
    except Exception:
        logger.warning("insightface not available — cross-video matching will be "
                       "appearance-only (same person in different clothes won't merge)")
        return {}

    try:
        app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        logger.warning(f"Face model init failed ({e}) — appearance-only matching")
        return {}

    results = json.load(open(f"outputs/runs/{vname}/reid_results.json", encoding="utf-8"))
    cap = cv2.VideoCapture(f"input/{vname}.mp4")
    by_sid = {}
    for fid_str, people in results.items():
        fid = int(fid_str)
        for p in people:
            sid = p.get("consolidated_id")
            if sid is None or sid == -1:
                continue
            by_sid.setdefault(sid, []).append((fid, p["bbox"]))

    desc = {}
    for sid, frames in by_sid.items():
        frames = sorted(frames)
        step = max(1, len(frames) // sample)
        feats = []
        for fid, bbox in frames[::step]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            head = frame[y1:y1 + int((y2 - y1) * 0.4) + 1, x1:x2]
            if head.size == 0:
                continue
            faces = app.get(head)
            if not faces:
                continue
            best = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            feats.append(best.normed_embedding)
        if feats:
            v = np.mean(feats, 0)
            desc[sid] = v / (np.linalg.norm(v) + 1e-8)
    cap.release()
    return desc


def build_global_ids(videos, device="cpu"):
    from reidentification.cross_camera_match import CrossCameraMatcher
    from reidentification.reid_main import ReIDEngine

    engine = ReIDEngine(device=device)

    camera_descriptors = {}
    camera_face = {}
    camera_results = {}
    for vname in videos:
        descriptors = _identity_descriptors(vname, engine)
        camera_descriptors[vname] = descriptors
        camera_face.update(
            {(vname, lid): f for lid, f in _identity_face_descriptors(vname).items()})
        camera_results[vname] = json.load(
            open(f"outputs/runs/{vname}/reid_results.json", encoding="utf-8"))
        logger.info(f"  {vname}: {sorted(descriptors.keys())} identity/identities")

    matcher = CrossCameraMatcher(face_descriptors=camera_face)
    for vname in videos:
        matcher.add_camera(vname, camera_descriptors[vname])

    # Cross-video similarity report (who matched whom).
    for vname in videos:
        for lid, desc in camera_descriptors[vname].items():
            gid = matcher.get_global_id(vname, lid)
            logger.info(f"  {vname} local_id={lid} -> global_id={gid}")

    combined = {}
    for vname in videos:
        cam_out = {}
        for fid_str, people in camera_results[vname].items():
            frame_out = []
            for p in people:
                p = dict(p)
                local_id = p.get("consolidated_id")
                gid = matcher.get_global_id(vname, local_id) \
                    if local_id not in (None, -1) else None
                p["global_id"] = gid
                p["name"] = None
                frame_out.append(p)
            cam_out[fid_str] = frame_out
        combined[vname] = cam_out

    combined["global_identities"] = {
        str(gid): {
            "cameras_seen_on": sorted({
                cam for (cam, _lid), g in matcher.local_to_global.items() if g == gid
            }),
        }
        for gid in matcher.global_descriptors().keys()
    }
    return combined, matcher


def render_global_videos(videos, combined_json_path):
    from utils import render_global_id_video
    for vname in videos:
        out_path = f"outputs/runs/{vname}/reid_global.mp4"
        ok = render_global_id_video(f"input/{vname}.mp4", combined_json_path, vname, out_path)
        logger.info(f"  {vname}: {'rendered' if ok else 'RENDER FAILED'} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Global IDs across videos + render")
    parser.add_argument("--videos", type=str, default="input1,input 2",
                        help="Comma-separated video names (must have run outputs)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--output", type=str, default="outputs/global/global_identities.json")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    videos = [v.strip() for v in args.videos.split(",")]

    combined, matcher = build_global_ids(videos, device=args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"✅ Global identities: {len(matcher.global_descriptors())} "
                f"across {len(videos)} video(s) -> {args.output}")

    if not args.skip_render:
        render_global_videos(videos, args.output)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
