"""
Search a video for registered people (face + body appearance).
=============================================================

Answers the question: "was registered person X in this video?"

Runs the same Re-ID engine the pipeline uses (so identity boundaries are
exactly what the project already produces), and alongside the frame loop it
collects a FACE GALLERY per stable identity from the person's head region
across the track. Each stable identity is then matched against every person
in the registration IdentityDatabase with score-level fusion of:

  - body appearance  (698-dim Re-ID descriptor, cosine similarity)
  - face             (128-dim face_recognition embedding, Euclidean-based
                      similarity via FaceCueExtractor)

Gait is the planned third cue — the same per-identity frame window this
module samples for faces is what a gait embedder will later consume.

Usage:
    python reidentification/video_search.py --video input/input\ 2.mp4 \
        --tracking "outputs/runs/input 2/tracking.json" --output search.json
"""

import argparse
import json
import logging
from pathlib import Path

import cv2

from registration.db_config import SEARCH_SETTINGS
from registration.identity_db import IdentityDatabase
from reidentification.reid_main import ReIDEngine

logger = logging.getLogger(__name__)


def search_registered_in_video(
    video_path,
    tracking_json_path,
    identity_db,
    device="cpu",
    output_json_path=None,
    sample_every=6,
    max_faces_per_track=30,
    face_upsample=3,
):
    """
    Args:
        video_path: input video file.
        tracking_json_path: DeepSORT tracking output (frame_id -> tracks).
        identity_db: registration.IdentityDatabase instance.
        device: 'cuda' or 'cpu'.
        output_json_path: optional path to write the search report.
        sample_every: extract gallery faces from every Nth frame per tracker
            (the engine already pays this face-detection cost internally, so
            sampling keeps the extra overhead small).
        max_faces_per_track: cap on gallery faces collected per tracker.
        face_upsample: dlib upsampling for gallery face detection. Surveillance
            faces are small (20-40px) and need 3-4 to be found reliably.

    Returns:
        {
          "video": str,
          "match_threshold": float,
          "identities": [{sid, faces_detected, matches}],
          "registered_persons": [{name, matched, best_score, appearance_sim,
                                  face_sim, cues, matched_identity}]
        }
    """
    from reidentification.face_cue import FaceCueExtractor

    engine = ReIDEngine(device=device)
    gallery_face_extractor = FaceCueExtractor(upsample_times=face_upsample)

    with open(tracking_json_path) as f:
        tracking_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    tid_faces = {}       # tracker id -> list of 128-dim face vectors
    tid_face_count = {}  # tracker id -> frames seen (for sampling)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        tracks = tracking_data.get(str(frame_id), [])
        frame_results = engine.process_frame(frame, tracks, frame_id)

        for p in frame_results:
            tid = p.get("id")
            # feature_dim == 0 marks a rejected/off-screen track — no usable
            # crop, so there is no face to extract.
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

        if frame_id % 30 == 0:
            logger.info(f"  Frame {frame_id:4d} | identities: {len(engine.identity_db)}")

    cap.release()
    engine.finalize_clustering()

    # Stable identity -> appearance descriptor + face gallery. engine.id_mapping
    # is the FINAL tracker->sid mapping (already reflects co-occurrence splits
    # and same-person merges), so faces collected under different tracker ids
    # that end on the same sid are pooled together.
    sid_info = {}
    for tid, sid in engine.id_mapping.items():
        faces = tid_faces.get(tid, [])
        if faces:
            sid_info.setdefault(sid, {"faces": []})["faces"].extend(faces)
    for sid, feat in engine.consolidated_features.items():
        sid_info.setdefault(sid, {})["appearance"] = feat

    threshold = SEARCH_SETTINGS["match_threshold"]

    identities = []
    for sid in sorted(sid_info.keys()):
        info = sid_info[sid]
        if "appearance" not in info:
            continue
        faces = info.get("faces", [])
        matches = identity_db.match_multimodal(info["appearance"], faces, top_k=10)
        identities.append({
            "sid": sid,
            "faces_detected": len(faces),
            "matches": matches,
        })
        logger.info(
            f"  Identity sid={sid}: faces={len(faces)} "
            f"-> {[m['name'] for m in matches[:3]]}"
        )

    # Aggregate per registered person across all identities.
    best_by_person = {}
    for sid, ident in zip([i["sid"] for i in identities], identities):
        for m in ident["matches"]:
            cur = best_by_person.get(m["name"])
            if cur is None or m["score"] > cur["score"]:
                best_by_person[m["name"]] = {
                    "score": m["score"],
                    "appearance_sim": m["appearance_sim"],
                    "face_sim": m["face_sim"],
                    "cues": m["cues"],
                    "sid": sid,
                }

    persons = []
    for name in identity_db.list_persons():
        m = best_by_person.get(name)
        persons.append({
            "name": name,
            "matched": m is not None and m["score"] >= threshold,
            "best_score": round(m["score"], 4) if m else None,
            "appearance_sim": m["appearance_sim"] if m else None,
            "face_sim": m["face_sim"] if m else None,
            "cues": m["cues"] if m else [],
            "matched_identity": m["sid"] if m else None,
        })

    report = {
        "video": str(video_path),
        "fps": fps,
        "frames": frame_id,
        "match_threshold": threshold,
        "identities": identities,
        "registered_persons": persons,
    }

    if output_json_path:
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"✅ Search report written -> {output_json_path}")

    return report


def _print_report(report):
    print("\n" + "=" * 60)
    print(f"  VIDEO SEARCH — {report['video']}")
    print(f"  threshold: {report['match_threshold']}")
    print("=" * 60)
    for p in report["registered_persons"]:
        status = "FOUND" if p["matched"] else "not found"
        print(f"  {p['name']:<20} {status:<10} score={p['best_score']} "
              f"face_sim={p['face_sim']} cues={','.join(p['cues'])}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(
        description="Search a video for registered persons (face + appearance)")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--tracking", required=True,
                        help="Tracking JSON from the pipeline")
    parser.add_argument("--output", default=None,
                        help="Where to write the search report JSON")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    args = parser.parse_args()

    db = IdentityDatabase()
    if len(db) == 0:
        print("⚠️  Identity database is empty — register someone first "
              "(registration/register_person.py or the web dashboard).")
        raise SystemExit(1)

    report = search_registered_in_video(
        args.video, args.tracking, db, device=args.device,
        output_json_path=args.output,
    )
    _print_report(report)
