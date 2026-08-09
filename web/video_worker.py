"""
Video processing worker.

Runs detect -> track -> re-ID -> registered-name matching -> render for a
single uploaded / webcam-recorded video. This module is designed to run in a
SEPARATE process (see single_video_job.py) so the dashboard server's event
loop never gets starved by the CPU/GPU-heavy pipeline work.

Every progress line is written to stdout so the parent can stream it into the
job console; the final result dict is what the caller persists.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Heavy model libraries load in THIS process only - the server stays light.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

from detection.detect_module import run_detection  # noqa: E402
from reidentification.reid_main import run_reid_pipeline  # noqa: E402
from tracking.track_module import run_tracking  # noqa: E402
from utils import render_global_id_video  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "web" / "outputs"
OUTPUTS_ROOT = ROOT_DIR / "outputs"
ALERTS_PATH = OUTPUTS_ROOT / "alerts.json"


def _artifact(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _flag_label(flag):
    from registration.identity_db import FLAGS
    return flag if flag in FLAGS else "normal"


def _read_alerts():
    if not ALERTS_PATH.exists():
        return []
    try:
        with open(ALERTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _append_alert(alert: dict):
    alerts = _read_alerts()
    alert.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    alerts.insert(0, alert)
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump(alerts[:200], f, indent=2, ensure_ascii=False)


def identify_video(input_path, tracking_path, reid_path, out_dir, label, camera_id="cam"):
    """
    Recognise registered persons in a processed video using the same
    face-vote + body-gate logic as the full multicam pipeline
    (face_name_resolver.filter_tracking_to_wanted). Returns a list of
    {"name", "flag", "similarity", "source", "track_id", "frames", "crop_url"}
    for every track that confidently matched a registered person.

    Non-normal flagged persons raise an alert event picked up by the Alerts panel.
    """
    import cv2

    from reidentification.face_name_resolver import filter_tracking_to_wanted
    from registration.face_db import FaceIdentityDB
    from registration.identity_db import IdentityDatabase

    try:
        with open(tracking_path, encoding="utf-8") as f:
            tracking = json.load(f)
        with open(reid_path, encoding="utf-8") as f:
            reid = json.load(f)
    except Exception as e:
        print(f"  WARN could not load tracking/reid for identification: {e}", flush=True)
        return [], {}

    tracking_by_cam = {camera_id: tracking}
    cameras = [{"camera_id": camera_id, "source": str(input_path)}]
    raw_consolidated = {camera_id: reid.get("consolidated_features", {})}

    track_cids = {}
    for frame_people in (reid.get("frames") or {}).values():
        for p in frame_people:
            cid = p.get("consolidated_id")
            if cid is not None and cid != -1:
                track_cids[(camera_id, str(p.get("id")))] = cid

    body_persons = None
    try:
        body_persons = IdentityDatabase().export_for_reid()
    except Exception as e:
        print(f"  WARN could not load registered body embeddings: {e}", flush=True)

    persons = []
    try:
        _, track_names = filter_tracking_to_wanted(
            tracking_by_cam, cameras,
            face_db=FaceIdentityDB(),
            track_cids=track_cids,
            raw_consolidated=raw_consolidated,
            body_persons=body_persons,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        print("  WARN wanted-person identification failed", flush=True)
        return persons, {}

    id_db = IdentityDatabase()
    frames = reid.get("frames") or {}
    cap = cv2.VideoCapture(str(input_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first_frame = {}
    for fid, people in sorted(frames.items(), key=lambda kv: int(kv[0])):
        for p in people:
            tid = str(p.get("id"))
            if tid not in first_frame:
                first_frame[tid] = int(fid)
    frame_buf = {}

    name_map = {}

    for (cam_id, tid), rec in track_names.items():
        rec = dict(rec)
        if not rec.get("name"):
            continue
        name = rec["name"]
        record = id_db.get_person(name) or {}
        flag = _flag_label((record.get("metadata") or {}).get("flag", "normal"))
        name_map[int(tid)] = {"name": name, "similarity": rec.get("similarity") or 0.0}

        crop_url = None
        fid = first_frame.get(str(tid))
        if fid is not None:
            try:
                if fid not in frame_buf:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
                    ok, fr = cap.read()
                    if ok:
                        frame_buf[fid] = fr
                fr = frame_buf.get(fid)
                if fr is not None:
                    for p in frames.get(str(fid), []):
                        if str(p.get("id")) == str(tid):
                            x1, y1, x2, y2 = map(int, p["bbox"])
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(fr.shape[1], x2), min(fr.shape[0], y2)
                            if x2 > x1 and y2 > y1:
                                crop_path = out_dir / f"crop_t{tid}.jpg"
                                cv2.imwrite(str(crop_path), fr[y1:y2, x1:x2])
                                crop_url = f"/outputs/{out_dir.name}/crop_t{tid}.jpg"
                            break
            except Exception as e:
                print(f"  WARN could not extract crop for track {tid}: {e}", flush=True)

        count = sum(1 for people in frames.values() if any(str(p.get("id")) == str(tid) for p in people))

        persons.append({
            "name": name,
            "flag": flag,
            "similarity": rec.get("similarity") or 0.0,
            "source": rec.get("source", "face"),
            "track_id": int(tid),
            "frames": count,
            "crop_url": crop_url,
            "camera_id": camera_id,
        })

        if flag != "normal":
            _append_alert({
                "person": name,
                "flag": flag,
                "source": label,
                "camera_id": camera_id,
                "track_id": int(tid),
                "frames": count,
                "similarity": rec.get("similarity") or 0.0,
                "crop_url": crop_url,
            })

    cap.release()
    persons.sort(key=lambda x: (-(x["similarity"] or 0), x["name"]))
    return persons, name_map


def _build_combined_frames(reid_path, name_map, camera_key="cam"):
    """Turn this video's Re-ID frames into the SAME combined-JSON schema the
    multi-camera pipeline writes (outputs/cross_camera/global_identities.json)
    so the existing renderer (utils.render_global_id_video -> draw_global_matches)
    draws the identical green boxes + names + confidence labels. Tracks that
    matched a registered person get `name`/`name_similarity` stamped on; the
    renderer skips everyone else - exactly the wanted-person gating of the
    multi-camera run."""
    with open(reid_path, encoding="utf-8") as f:
        reid = json.load(f)
    frames = reid.get("frames") or {}
    combined = {camera_key: {}}
    for fid, people in frames.items():
        out_people = []
        for p in people:
            entry = dict(p)
            try:
                tid = int(p.get("id"))
            except (TypeError, ValueError):
                tid = None
            rec = name_map.get(tid) if tid is not None else None
            if rec:
                entry["name"] = rec["name"]
                entry["name_similarity"] = rec["similarity"]
                entry["global_id"] = tid
            out_people.append(entry)
        combined[camera_key][str(fid)] = out_people
    return combined


def run_video_job(job_id, input_path, label, device, render=True, video_hash="", camera_id="cam"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Artifacts are named from the video's content hash so a later cache hit can
    # reuse the exact same rendered output + intermediate JSON files.
    stem = video_hash[:12] if video_hash else job_id
    detections_path = OUTPUT_DIR / f"{stem}_detections.json"
    tracking_path = OUTPUT_DIR / f"{stem}_tracking.json"
    reid_path = OUTPUT_DIR / f"{stem}_reid.json"
    out_video = OUTPUT_DIR / f"{stem}_identified.mp4"
    crops_dir = OUTPUT_DIR / f"crops_{stem}"

    print(f"Processing '{label}'...", flush=True)
    print("STEP 1 · detecting persons...", flush=True)
    run_detection(
        str(input_path), str(detections_path),
        conf_threshold=0.6, min_height=50, min_area_ratio=0.001, device=device,
    )
    if not _artifact(detections_path):
        raise RuntimeError("Detection output not created")
    print(f"  saved detections -> {detections_path.name}", flush=True)

    print("STEP 2 · tracking persons...", flush=True)
    run_tracking(str(input_path), str(detections_path), str(tracking_path))
    if not _artifact(tracking_path):
        raise RuntimeError("Tracking output not created")
    print(f"  saved tracking -> {tracking_path.name}", flush=True)

    print("STEP 3 · re-identifying (appearance + face cues)...", flush=True)
    run_reid_pipeline(str(input_path), str(tracking_path), str(reid_path), device=device)
    if not _artifact(reid_path):
        raise RuntimeError("Re-ID output not created")
    print(f"  saved reid -> {reid_path.name}", flush=True)

    print("STEP 4 · matching against registered persons...", flush=True)
    persons, name_map = identify_video(input_path, tracking_path, reid_path, crops_dir, label, camera_id)
    for p in persons:
        print(f"  identified: {p['name']} (similarity {p['similarity']:.3f}, "
              f"{p['frames']} frame(s), via {p['source']})", flush=True)

    result = {
        "label": label,
        "camera_id": camera_id,
        "persons": persons,
        "alert_count": sum(1 for p in persons if p["flag"] != "normal"),
        "output_url": None,
        "output_name": None,
    }

    if render:
        print("STEP 5 · rendering output video...", flush=True)
        combined_path = OUTPUT_DIR / f"{stem}_combined.json"
        combined_path.write_text(
            json.dumps(_build_combined_frames(str(reid_path), name_map, camera_id)),
            encoding="utf-8",
        )
        final_path = render_global_id_video(
            str(input_path), str(combined_path), camera_id, str(out_video)
        )
        if final_path and _artifact(Path(final_path)):
            result["output_url"] = f"/outputs/{Path(final_path).name}"
            result["output_name"] = Path(final_path).name
        elif not final_path or not Path(final_path).exists():
            raise RuntimeError("Rendered video not created")

    return result


def main():
    if len(sys.argv) < 5:
        print("usage: single_video_job.py <job_id> <input_path> <label> <device> [video_hash] [camera_id]", flush=True)
        sys.exit(2)
    job_id, input_path, label, device = sys.argv[1:5]
    video_hash = sys.argv[5] if len(sys.argv) > 5 else ""
    camera_id = sys.argv[6] if len(sys.argv) > 6 else "cam"
    try:
        result = run_video_job(job_id, input_path, label, device, video_hash=video_hash, camera_id=camera_id)
        result_path = OUTPUT_DIR / f"{job_id}_result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        print("JOB_DONE", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"JOB_ERROR {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
