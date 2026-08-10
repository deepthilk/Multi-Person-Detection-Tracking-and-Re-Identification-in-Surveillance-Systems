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
  - face             (512-dim ArcFace embedding via InsightFace, cosine
                      similarity)

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

# An appearance-only match is reported only when the identity's top
# appearance pick beats its runner-up by at least this much. Under identical
# uniforms a face-less identity scores 0.6-0.8 against EVERYONE — there is
# no way to tell who it is, so a narrow spread must not count as a hit.
APPEARANCE_ONLY_MARGIN = 0.08


def search_registered_in_video(
    video_path,
    tracking_json_path,
    identity_db,
    device="cpu",
    output_json_path=None,
    render_video_path=None,
    sample_every=6,
    max_faces_per_track=30,
    face_upsample=3,
    stride=1,
):
    """
    Args:
        video_path: input video file.
        tracking_json_path: DeepSORT tracking output (frame_id -> tracks).
        identity_db: registration.IdentityDatabase instance.
        device: 'cuda' or 'cpu'.
        output_json_path: optional path to write the search report.
        render_video_path: optional path to write an overlay video — every
            detected identity is boxed and labelled with the name of the
            registered person it matched (or "Unknown n" when no match).
        sample_every: accepted for backward compatibility but ignored — every
            processed frame contributes a gallery face (stride thins the cost).
        max_faces_per_track: cap on gallery faces collected per tracker.
        face_upsample: accepted for backward compatibility but ignored —
            the InsightFace detector sizes input via det_size.
        stride: process only every Nth frame. Same person appears across many
            consecutive frames, so identity results barely change while runtime
            drops ~stride× — the main CPU-only speed knob.

    Returns:
        {
          "video": str,
          "match_threshold": float,
          "identities": [{sid, faces_detected, matches}],
          "registered_persons": [{name, matched, best_score, appearance_sim,
                                  face_sim, cues, matched_identity}]
        }
    """
    from reidentification.insight_face import InsightFaceExtractor

    engine = ReIDEngine(device=device)
    gallery_face_extractor = InsightFaceExtractor()

    with open(tracking_json_path) as f:
        tracking_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    tid_faces = {}       # tracker id -> list of 512-dim face vectors
    frame_people = {}    # frame_id -> [(bbox, tid)] for overlay rendering

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

        for p in frame_results:
            tid = p.get("id")
            # feature_dim == 0 marks a rejected/off-screen track — no usable
            # crop, so there is no face to extract.
            if tid is None or p.get("feature_dim", 0) == 0:
                continue
            frame_people.setdefault(frame_id, []).append((p["bbox"], tid))
            # Keep a face from EVERY processed frame (stride already thins the
            # cost), capped by max_faces_per_track. Modulo-thinning on top of
            # stride is what previously starved borderline matches: a real
            # same-person face at 0.53 (vs 0.45 threshold) was sampled out and
            # the person came back "not found" even though she was in the video.
            if len(tid_faces.get(tid, [])) >= max_faces_per_track:
                continue
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

    # Aggregate per registered person across all identities. Prefer
    # FACE-CONFIRMED evidence: a blurry-looking identity can match everyone by
    # body appearance alone (identical uniforms), and those noisy appearance
    # scores often outrank the genuine face-confirmed hit. So:
    #   1. any face-confirmed match beats every appearance-only match;
    #   2. an appearance-only match is only accepted when it is the CLEAR
    #      winner for its identity (top-ranked and beating the runner-up by a
    #      margin) — a face-less identity that matches several people at
    #      similar appearance similarity cannot identify anyone.
    best_by_person = {}
    per_ident_matches = {i["sid"]: i["matches"] for i in identities}
    for sid, ident in zip([i["sid"] for i in identities], identities):
        for m in ident["matches"]:
            face_confirmed = "face" in m["cues"] and m["face_sim"] is not None
            cur = best_by_person.get(m["name"])
            cur_face = cur and "face" in cur["cues"] and cur["face_sim"] is not None
            if face_confirmed and not cur_face:
                best_by_person[m["name"]] = {**m, "sid": sid}
            elif face_confirmed and cur_face:
                if m["score"] > cur["score"]:
                    best_by_person[m["name"]] = {**m, "sid": sid}
            elif not face_confirmed and not cur_face:
                # appearance-only: only the identity's clear top pick counts
                top = per_ident_matches[sid][0] if per_ident_matches[sid] else None
                if top and top["name"] == m["name"]:
                    runner = per_ident_matches[sid][1] if len(per_ident_matches[sid]) > 1 else None
                    margin = top["score"] - (runner["score"] if runner else 0.0)
                    if margin >= APPEARANCE_ONLY_MARGIN:
                        if cur is None or m["score"] > cur["score"]:
                            best_by_person[m["name"]] = {**m, "sid": sid}

    persons = []
    for name in identity_db.list_persons():
        m = best_by_person.get(name)
        # match_multimodal already filters by the branch-appropriate threshold
        # (0.55 appearance-only / ambiguous, 0.45 face-confirmed, face-veto
        # never returned), so a returned match IS a match — re-applying the
        # plain appearance threshold here would wrongly downgrade a
        # face-confirmed hit like v3 sid=2 (score 0.52, face 0.535 >= 0.45).
        persons.append({
            "name": name,
            "matched": m is not None,
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
        "stride": stride,
        "match_threshold": threshold,
        "identities": identities,
        "registered_persons": persons,
    }

    if output_json_path:
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"✅ Search report written -> {output_json_path}")

    if render_video_path:
        sid_label = {i["sid"]: i["matches"][0]["name"]
                     for i in identities if i["matches"]}
        _render_search_video(video_path, frame_people, engine.id_mapping,
                             sid_label, render_video_path, fps)

    return report


def _render_search_video(video_path, frame_people, tid_to_sid, sid_label,
                         output_video_path, fps):
    """Draw a box + name label per detected identity onto every frame and
    write an mp4 (re-encoded with ffmpeg for browser playback)."""
    from utils import _reencode_for_browser

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Render: failed to open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not fps or fps <= 1:
        fps = 30

    out = None
    for codec in ("mp4v", "avc1", "H264"):
        writer = cv2.VideoWriter(output_video_path,
                                 cv2.VideoWriter_fourcc(*codec), fps,
                                 (width, height))
        if writer.isOpened():
            out = writer
            break
    if out is None:
        cap.release()
        logger.error("Render: failed to initialize VideoWriter")
        return

    # Number unknown identities Unknown-Id1, Unknown-Id2, ... by first
    # appearance (instead of exposing the internal stable-id).
    known_sids = set(sid_label.keys())
    first_appear = {}
    for fid in sorted(frame_people.keys()):
        for _bbox, tid in frame_people[fid]:
            sid = tid_to_sid.get(tid)
            if sid is not None and sid not in first_appear:
                first_appear[sid] = fid
    unknown_label = {}
    idx = 0
    unknown_sids = sorted(
        {s for s in tid_to_sid.values() if s is not None and s not in known_sids},
        key=lambda s: first_appear.get(s, float("inf")),
    )
    for sid in unknown_sids:
        idx += 1
        unknown_label[sid] = f"Unknown-Id{idx}"

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        for bbox, tid in frame_people.get(frame_id, []):
            sid = tid_to_sid.get(tid)
            if sid is None:
                continue
            name = sid_label.get(sid)
            if name is None:
                name = unknown_label.get(sid, f"Unknown {sid}")
            color = (0, 255, 0) if sid in known_sids else (0, 0, 255)
            label = name if name else f"Unknown {sid}"
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        out.write(frame)

    cap.release()
    out.release()
    _reencode_for_browser(output_video_path)
    logger.info(f"✅ Rendered search video -> {output_video_path}")


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
    parser.add_argument("--render", default=None,
                        help="Where to write the overlay video (box + name per person)")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame (default 1 = all frames). "
                             "Higher = faster, slightly coarser.")
    args = parser.parse_args()

    db = IdentityDatabase()
    if len(db) == 0:
        print("⚠️  Identity database is empty — register someone first "
              "(registration/register_person.py or the web dashboard).")
        raise SystemExit(1)

    report = search_registered_in_video(
        args.video, args.tracking, db, device=args.device,
        output_json_path=args.output, render_video_path=args.render,
        stride=args.stride,
    )
    _print_report(report)
