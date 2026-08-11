from pathlib import Path
import hashlib
import json
import logging
import shutil
import sys
import time
import uuid
from typing import Dict, List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from detection.detect_module import run_detection
from tracking.track_module import run_tracking
from reidentification.reid_main import run_reid_pipeline
from registration.identity_db import IdentityDatabase
from registration.register_person import register_person
from utils import render_reid_video, extract_track_thumbnail

from multicam_pipeline import run_camera_reid
from cross_camera import unify_session_files

logger = logging.getLogger("web.server")

WEB_DIR = ROOT_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
UPLOAD_DIR = WEB_DIR / "uploads"
OUTPUT_DIR = WEB_DIR / "outputs"
REG_UPLOAD_DIR = WEB_DIR / "uploads" / "registration"
# where registration/register_person.py persists a copy of every registered
# photo (see registration/db_config.py -> DB_SETTINGS["images_dir"]) — this
# module only reads from it, never writes, matching the "reuse, never edit"
# pattern the rest of the registration integration follows.
REG_IMAGES_DIR = ROOT_DIR / "outputs" / "registration" / "images"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Multi-Person Detection & Re-ID — Web Console")

# single-video jobs (legacy quick-test panel)
JOBS: Dict[str, dict] = {}
# multi-camera sessions: session_id -> {"cameras": {camera_id: job_dict}}
SESSIONS: Dict[str, dict] = {}

# ── operations telemetry: rolling in-memory buffers, capped so a long
#    session can't grow these unbounded ──────────────────────────────────
LOGS: List[dict] = []
LATENCY_SAMPLES: List[dict] = []
_MAX_LOGS = 500
_MAX_LATENCY_SAMPLES = 200


def _log(level: str, message: str, camera_id: str = None):
    entry = {
        "ts": time.time(),
        "level": level,  # "info" | "warn" | "critical"
        "message": message,
        "camera_id": camera_id,
    }
    LOGS.append(entry)
    if len(LOGS) > _MAX_LOGS:
        del LOGS[: len(LOGS) - _MAX_LOGS]
    logger.info("[%s] %s", level.upper(), message)


def _record_latency(stage: str, seconds: float, camera_id: str = None):
    LATENCY_SAMPLES.append(
        {"ts": time.time(), "stage": stage, "seconds": round(seconds, 3), "camera_id": camera_id}
    )
    if len(LATENCY_SAMPLES) > _MAX_LATENCY_SAMPLES:
        del LATENCY_SAMPLES[: len(LATENCY_SAMPLES) - _MAX_LATENCY_SAMPLES]


# ══════════════════════════════════════════════════════════════════════════
#  Watch-list alerts (flagged-person sightings)
# ══════════════════════════════════════════════════════════════════════════

# where watch-list alert events are persisted; also owned here:
# - dismissed_alerts.json (ids of alerts the operator marked "handled")
ALERTS_PATH = ROOT_DIR / "outputs" / "alerts.json"
DISMISSED_PATH = ROOT_DIR / "outputs" / "dismissed_alerts.json"


def _read_alerts() -> List[dict]:
    if not ALERTS_PATH.exists():
        return []
    try:
        with open(ALERTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"Could not read alerts file: {e}")
        return []


def _append_alert(alert: dict):
    """Persist an alert event to the top of alerts.json (capped at 200).
    De-duplicates by (person, camera_id, track_id) so re-runs / cross-camera
    corrections don't stack identical rows."""
    alerts = _read_alerts()
    key = (alert.get("person"), alert.get("camera_id"), alert.get("track_id"))
    alerts = [a for a in alerts
              if (a.get("person"), a.get("camera_id"), a.get("track_id")) != key]
    alert.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    alerts.insert(0, alert)
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump(alerts[:200], f, indent=2, ensure_ascii=False)


def _alert_event_id(ev: dict) -> str:
    """Stable id for a persisted alert event (alerts.json)."""
    key = {
        "person": ev.get("person"),
        "source": ev.get("source"),
        "time": ev.get("time"),
        "camera_id": ev.get("camera_id"),
        "track_id": ev.get("track_id"),
    }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _read_dismissed() -> set:
    try:
        if DISMISSED_PATH.exists():
            data = json.loads(DISMISSED_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(data)
    except Exception as e:
        logger.warning(f"Could not read dismissed alerts: {e}")
    return set()


def _write_dismissed(dismissed: set):
    DISMISSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        DISMISSED_PATH.write_text(
            json.dumps(sorted(dismissed), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"Could not write dismissed alerts: {e}")


def _flag_label(flag):
    from registration.identity_db import FLAGS
    return flag if flag in FLAGS else "normal"


def _raise_flag_alerts(cam: dict):
    """For one finished camera, persist an alert for every recognised track
    whose registered person carries a non-normal watch-list flag."""
    for p in cam.get("people", []):
        name = p.get("name")
        if not name:
            continue
        record = _get_db().get_person(name) or {}
        meta = record.get("metadata", {})
        flag = _flag_label(meta.get("flag", "normal"))
        if flag == "normal":
            continue
        _append_alert({
            "person": name,
            "flag": flag,
            "details": meta.get("details", ""),
            "source": cam.get("label", cam.get("camera_id", "cam")),
            "camera_id": cam.get("camera_id"),
            "track_id": p.get("track_id"),
            "similarity": round(float(p["similarity"] or 0.0), 3),
            "crop_url": p.get("thumb_url"),
            "first_seen_sec": p.get("first_seen_sec"),
            "last_seen_sec": p.get("last_seen_sec"),
        })


def _ensure_artifact(path: Path, label: str):
    if not path.exists():
        raise RuntimeError(f"{label} not created")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{label} is empty")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/reg-photos", StaticFiles(directory=REG_IMAGES_DIR), name="reg-photos")


@app.middleware("http")
async def no_cache_static(request, call_next):
    """Browsers otherwise serve stale app.js/styles.css (e.g. the old blinking
    box rendering) and stale re-rendered videos/thumbnails after corrections."""
    response = await call_next(request)
    if request.url.path.startswith(("/static/", "/outputs/", "/reg-photos/")):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


# ══════════════════════════════════════════════════════════════════════════
#  2 & 3 — Person Registration + Identity Database
# ══════════════════════════════════════════════════════════════════════════

def _get_db() -> IdentityDatabase:
    return IdentityDatabase()


def _photo_url(path_str: str) -> str:
    # image_paths are stored like "outputs/registration/images/<name>/<file>";
    # taking the last two path parts is robust whether the stored string is
    # relative or absolute, and avoids depending on the exact separator.
    parts = Path(path_str).parts[-2:]
    return "/reg-photos/" + "/".join(parts)


def _person_summary(name: str, record: dict) -> dict:
    meta = record.get("metadata", {})
    image_paths = meta.get("image_paths", [])
    return {
        "name": name,
        "num_images": meta.get("num_images", 0),
        "registered_at": meta.get("registered_at"),
        "last_updated": meta.get("last_updated"),
        "image_paths": image_paths,
        "photos": [_photo_url(p) for p in image_paths],
        "person_id": meta.get("person_id"),
        "flag": meta.get("flag", "normal"),
        "details": meta.get("details", ""),
    }


@app.get("/api/persons")
def list_persons():
    db = _get_db()
    return JSONResponse([_person_summary(n, db.get_person(n)) for n in db.list_persons()])


@app.get("/api/persons/search")
def search_persons(q: str = ""):
    db = _get_db()
    if not q.strip():
        return JSONResponse([_person_summary(n, db.get_person(n)) for n in db.list_persons()])
    return JSONResponse([_person_summary(n, rec) for n, rec in db.search_by_name(q)])


@app.post("/api/persons")
def add_person(name: str = Form(...), files: List[UploadFile] = File(...),
               person_id: str = Form(""), flag: str = Form("normal"),
               details: str = Form("")):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not files:
        raise HTTPException(status_code=400, detail="At least one photo is required")

    db = _get_db()
    if name in db.list_persons():
        raise HTTPException(status_code=409, detail=f"'{name}' is already registered — use edit to add more photos")

    saved_paths = _save_uploads(name, files)
    try:
        record = register_person(name, saved_paths, db=db,
                                 person_id=person_id.strip() or None,
                                 flag=flag, details=details.strip())
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return JSONResponse(_person_summary(name, record), status_code=201)


@app.put("/api/persons/{name}/images")
def add_person_images(name: str, files: List[UploadFile] = File(...)):
    db = _get_db()
    if name not in db.list_persons():
        raise HTTPException(status_code=404, detail=f"'{name}' is not registered")
    if not files:
        raise HTTPException(status_code=400, detail="At least one photo is required")

    saved_paths = _save_uploads(name, files)
    try:
        record = register_person(name, saved_paths, db=db)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return JSONResponse(_person_summary(name, record))


@app.put("/api/persons/{name}/rename")
def rename_person(name: str, new_name: str = Form(...)):
    new_name = new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New name is required")

    db = _get_db()
    record = db.get_person(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"'{name}' is not registered")
    if new_name != name and new_name in db.list_persons():
        raise HTTPException(status_code=409, detail=f"'{new_name}' is already registered")

    # IdentityDatabase has no rename() — this module never edits identity_db.py,
    # so we move the record via the same JSON store it already owns and saves.
    db._data[new_name] = record
    if new_name != name:
        del db._data[name]
    db.save()
    return JSONResponse(_person_summary(new_name, record))


@app.delete("/api/persons/{name}")
def delete_person(name: str):
    db = _get_db()
    if not db.delete_person(name):
        raise HTTPException(status_code=404, detail=f"'{name}' is not registered")
    return JSONResponse({"deleted": name})


@app.put("/api/persons/{name}/profile")
def update_person_profile(name: str, person_id: str = Form(""), flag: str = Form("normal"),
                          details: str = Form("")):
    """Update a registered person's watch-list profile (case ID / flag / notes)
    without touching their embeddings or photos."""
    db = _get_db()
    if db.get_person(name) is None:
        raise HTTPException(status_code=404, detail=f"'{name}' is not registered")
    flag = _flag_label(flag)
    try:
        db.update_metadata(
            name,
            person_id=person_id.strip() or None,
            flag=flag,
            details=details.strip(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    _log("info", f"Profile updated for '{name}' (flag={flag})")
    return JSONResponse(_person_summary(name, db.get_person(name)))


def _save_uploads(name: str, files: List[UploadFile]) -> List[str]:
    dest_dir = REG_UPLOAD_DIR / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, f in enumerate(files):
        safe_name = Path(f.filename or f"photo_{i}.jpg").name
        dest = dest_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        with dest.open("wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved.append(str(dest))
    return saved


# ══════════════════════════════════════════════════════════════════════════
#  1, 4 & 5 — Multi-Camera Upload, Processing Status, Results
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/session/start")
def start_session(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    labels: List[str] = Form(...),
    stride: int = Form(3),
    det_stride: int = Form(1),
):
    if not files:
        raise HTTPException(status_code=400, detail="No camera videos uploaded")
    if len(labels) != len(files):
        raise HTTPException(status_code=400, detail="Every camera upload needs a label")
    stride = min(max(int(stride), 1), 10)
    det_stride = min(max(int(det_stride), 1), 10)

    session_id = uuid.uuid4().hex[:10]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cameras: Dict[str, dict] = {}

    for i, (f, label) in enumerate(zip(files, labels)):
        camera_id = f"cam{i + 1}"
        safe_name = Path(f.filename or f"{camera_id}.mp4").name
        input_path = UPLOAD_DIR / f"{session_id}_{camera_id}_{safe_name}"
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        cameras[camera_id] = {
            "camera_id": camera_id,
            "label": label or camera_id,
            "input_path": str(input_path),
            "status": "queued",
            "percent": 0,
            "message": "Queued",
            "output_url": None,
            "people": [],
            "fps": None,
            "frame_drop_rate": None,
            "reid_json_path": None,
            "stride": stride,
        }

        background_tasks.add_task(
            _run_camera_job,
            session_id,
            camera_id,
            input_path,
            device,
            stride,
            det_stride,
        )

    SESSIONS[session_id] = {"cameras": cameras}
    # Runs AFTER every camera job (FastAPI BackgroundTasks are sequential),
    # unifying identities that appear on more than one camera.
    background_tasks.add_task(_unify_session, session_id)
    return JSONResponse({"session_id": session_id, "cameras": list(cameras.keys())})


@app.get("/api/session/{session_id}/progress")
def session_progress(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    cams = list(session["cameras"].values())
    overall = int(sum(c["percent"] for c in cams) / max(1, len(cams)))
    return JSONResponse({"cameras": cams, "overall_percent": overall, "unified": session.get("unified", False)})


@app.get("/api/session/{session_id}/results")
def session_results(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    cams = session["cameras"]
    matched_by_name: Dict[str, dict] = {}
    unmatched: List[dict] = []
    total_people = 0

    for cam in cams.values():
        for person in cam.get("people", []):
            total_people += 1
            sighting = {
                "camera_id": cam["camera_id"],
                "camera_label": cam["label"],
                "track_id": person["track_id"],
                "global_id": person.get("global_id"),
                "first_seen_sec": person["first_seen_sec"],
                "last_seen_sec": person["last_seen_sec"],
                "thumb_url": person.get("thumb_url"),
            }
            if person.get("name"):
                entry = matched_by_name.setdefault(
                    person["name"], {"name": person["name"], "similarity": person["similarity"], "sightings": []}
                )
                entry["sightings"].append(sighting)
                entry["similarity"] = max(entry["similarity"] or 0, person["similarity"] or 0)
            else:
                unmatched.append(
                    {
                        "track_id": f"{cam['camera_id']}-{person['track_id']}",
                        **sighting,
                    }
                )

    matched = sorted(matched_by_name.values(), key=lambda m: m["name"].lower())
    summary = {
        "camera_count": len(cams),
        "people_detected": total_people,
        "matched": len(matched),
        "unknown": len(unmatched),
        "all_cameras_done": all(c["status"] in ("completed", "error") for c in cams.values()),
    }
    cameras = [
        {
            "camera_id": c["camera_id"],
            "label": c["label"],
            "output_url": c.get("output_url"),
            "people": [
                {
                    "track_id": p["track_id"],
                    "name": p.get("name"),
                    "similarity": p.get("similarity"),
                    "face_sim": p.get("face_sim"),
                    "global_id": p.get("global_id"),
                }
                for p in c.get("people", [])
            ],
        }
        for c in cams.values()
    ]
    return JSONResponse({"summary": summary, "matched": matched, "unmatched": unmatched, "cameras": cameras})


def _rewrite_tracks(reid_path: Path, corrected: dict):
    """Write corrected names/evidence back into a camera's reid.json
    __tracks__ section so re-rendered videos and future reads use them."""
    if not reid_path.exists():
        return
    with open(reid_path, encoding="utf-8") as f:
        data = json.load(f)
    tracks = data.get("__tracks__", {})
    for tid, c in corrected.items():
        if tid in tracks:
            tracks[tid]["name"] = c.get("name")
            tracks[tid]["similarity"] = c.get("similarity")
            tracks[tid]["face_sim"] = c.get("face_sim")
            tracks[tid]["cues"] = c.get("cues", [])
            tracks[tid]["manual"] = bool(c.get("manual")) or tracks[tid].get("manual", False)
            if c.get("global_id") is not None:
                tracks[tid]["global_id"] = c["global_id"]
    with open(reid_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _re_render(cam: dict):
    """Re-render a camera's annotated video from its (possibly corrected)
    reid.json, so on-screen labels reflect the latest names."""
    reid_path = Path(cam["reid_json_path"])
    out = reid_path.with_name(reid_path.name.replace("_reid.json", "_reid.mp4"))
    try:
        render_reid_video(cam["input_path"], str(reid_path), str(out))
    except Exception:
        logger.exception("Re-render failed for %s", cam["camera_id"])


def _unify_session(session_id: str):
    """After every camera finishes, unify identities across cameras using the
    face galleries persisted in each reid.json, then re-render with the
    resolved names."""
    session = SESSIONS.get(session_id)
    if not session:
        return
    cams = session["cameras"]
    completed = {
        cid: cam
        for cid, cam in cams.items()
        if cam.get("status") == "completed" and cam.get("reid_json_path")
    }
    if not completed:
        session["unified"] = True
        return
    reid_paths = {cid: Path(cam["reid_json_path"]) for cid, cam in completed.items()}
    try:
        corrected = unify_session_files(reid_paths)
    except Exception:
        logger.exception("Cross-camera unify failed for session %s", session_id)
        session["unified"] = True
        return

    changes = 0
    for cid, cam in completed.items():
        corr = corrected.get(cid, {})
        for p in cam.get("people", []):
            c = corr.get(str(p.get("track_id")))
            if not c:
                continue
            if c["name"] != p.get("name"):
                changes += 1
                _log("info", f"{cam['label']}: cross-camera → track {p['track_id']} = '{c['name']}'")
            p["name"] = c["name"]
            p["similarity"] = c["similarity"]
            p["face_sim"] = c["face_sim"]
            p["cues"] = c.get("cues", [])
            p["global_id"] = c["global_id"]
        _rewrite_tracks(Path(cam["reid_json_path"]), corr)
        _re_render(cam)
        # A cross-camera resolution may have just named a flagged person that
        # the single-camera pass left unknown — raise (deduped) alerts again.
        _raise_flag_alerts(cam)

    if changes:
        _log("info", f"Cross-camera unification: {changes} name change(s) across {len(completed)} camera(s)")
    else:
        _log("info", "Cross-camera unification: all cameras already agree")

    session["unified"] = True


@app.post("/api/session/{session_id}/correct")
def correct_track(session_id: str, camera_id: str = Form(...), track_id: int = Form(...),
                  name: str = Form("")):
    """Manual correction: assign a track a name (registered person or a brand
    new one) or clear it back to 'unknown'. Persists the track's face +
    appearance evidence into the identity DB so FUTURE videos of that person
    resolve automatically too, then re-propagates cross-camera and re-renders.
    """
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    cam = session["cameras"].get(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not cam.get("reid_json_path"):
        raise HTTPException(status_code=409, detail="This camera hasn't finished processing yet")

    reid_path = Path(cam["reid_json_path"])
    with open(reid_path, encoding="utf-8") as f:
        data = json.load(f)
    track = (data.get("__tracks__") or {}).get(str(track_id))
    if track is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found for {camera_id}")

    name = name.strip()
    faces = track.get("faces") or []
    mean = track.get("mean_feature")

    if name and not mean:
        raise HTTPException(status_code=422, detail="This track has no appearance descriptor — can't register it")

    sim = face_sim = None
    cues = []
    if name:
        db = _get_db()
        # Evidence measured against the EXISTING gallery (before this track's
        # own faces join it, which would trivially self-match at ~1.0).
        pre = db.match_multimodal(mean, [list(f) for f in faces], top_k=1)
        match = next((m for m in pre if m["name"] == name), None)
        if match:
            sim, face_sim, cues = match["score"], match["face_sim"], match["cues"]
        # Persist the track's evidence ONLY where it corroborates the person's
        # existing gallery. A corrected track may be a false merge or a
        # mis-named track, and blindly appending every face used to poison the
        # DB with other people's faces (then self-confirming at ~1.0 later).
        db.add_person_corroborated(name, [mean], face_embeddings=faces)

    for p in cam.get("people", []):
        if p.get("track_id") == track_id:
            p["name"] = name or None
            p["similarity"] = sim
            p["face_sim"] = face_sim
            p["cues"] = cues
            p["manual"] = bool(name)
            break

    track["name"] = name or None
    track["similarity"] = sim
    track["face_sim"] = face_sim
    track["cues"] = cues
    track["manual"] = bool(name)
    with open(reid_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    _re_render(cam)
    _log("info", f"{cam['label']}: manual correction → track {track_id} = "
                 f"{name!r} ({'face=' + str(round(face_sim, 3)) if face_sim else 'manual'})")
    # A manual correction can be what first reveals a flagged person's identity.
    _raise_flag_alerts(cam)

    # Propagate the corrected name to the same person on other cameras.
    if len(session["cameras"]) > 1:
        _unify_session(session_id)

    return JSONResponse({"ok": True, "people": cam.get("people", [])})


def _run_camera_job(session_id: str, camera_id: str, input_path: Path, device: str, stride: int = 3, det_stride: int = 2):
    cam = SESSIONS[session_id]["cameras"][camera_id]
    detections_path = OUTPUT_DIR / f"{session_id}_{camera_id}_detections.json"
    tracking_path = OUTPUT_DIR / f"{session_id}_{camera_id}_tracking.json"
    reid_path = OUTPUT_DIR / f"{session_id}_{camera_id}_reid.json"
    output_video_path = OUTPUT_DIR / f"{session_id}_{camera_id}_reid.mp4"
    label = cam.get("label", camera_id)

    _log("info", f"{label}: job queued on {device.upper()}", camera_id)

    try:
        t0 = time.time()
        cam.update({"status": "running", "percent": 10, "message": "Detecting people"})
        run_detection(
            str(input_path),
            str(detections_path),
            conf_threshold=0.6,
            min_height=50,
            min_area_ratio=0.001,
            device=device,
            stride=det_stride,
        )
        _ensure_artifact(detections_path, "Detection output")
        _record_latency("detection", time.time() - t0, camera_id)
        _log("info", f"{label}: detection complete ({time.time() - t0:.1f}s)", camera_id)

        t1 = time.time()
        cam.update({"percent": 35, "message": "Tracking across frames"})
        run_tracking(str(input_path), str(detections_path), str(tracking_path))
        _ensure_artifact(tracking_path, "Tracking output")
        _record_latency("tracking", time.time() - t1, camera_id)
        _log("info", f"{label}: tracking complete ({time.time() - t1:.1f}s)", camera_id)

        t2 = time.time()
        cam.update({"percent": 60, "message": "Matching against known people"})
        db = _get_db()
        summary = run_camera_reid(
            str(input_path),
            str(tracking_path),
            str(reid_path),
            device=device,
            identity_db=db,
            stride=stride,
        )
        _ensure_artifact(reid_path, "Re-ID output")
        _record_latency("reid", time.time() - t2, camera_id)

        cam["people"] = summary["people"]
        cam["fps"] = summary["fps"]
        cam["frame_drop_rate"] = summary["frame_drop_rate"]
        cam["reid_json_path"] = str(reid_path)

        # One screenshot per track, taken from the video itself (largest box),
        # so the Results tab shows WHO each row is talking about.
        for p in cam["people"]:
            thumb_path = OUTPUT_DIR / f"{session_id}_{camera_id}_t{p['track_id']}.jpg"
            if extract_track_thumbnail(str(input_path), str(reid_path), p["track_id"], thumb_path):
                p["thumb_url"] = f"/outputs/{thumb_path.name}"

        # Watch-list alert for every flagged registered person recognised here.
        _raise_flag_alerts(cam)

        if summary["frame_drop_rate"] > 0.02:
            _log(
                "warn",
                f"{label}: {summary['frame_drop_rate'] * 100:.1f}% frame drop detected during decode",
                camera_id,
            )

        for p in summary["people"]:
            if p["name"]:
                level = "critical" if p["similarity"] and p["similarity"] >= 0.75 else "info"
                tag = "CRITICAL RE-ID MATCH" if level == "critical" else "MATCH"
                conf = f"{p['similarity'] * 100:.1f}%" if p["similarity"] is not None else "n/a"
                _log(
                    level,
                    f"{label}: [{tag}] track {p['track_id']} resolved to '{p['name']}' "
                    f"(confidence {conf})",
                    camera_id,
                )
            else:
                _log("info", f"{label}: track {p['track_id']} has no identity match — unknown", camera_id)

        cam.update({"percent": 90, "message": "Rendering annotated video"})
        t3 = time.time()
        if render_reid_video(str(input_path), str(reid_path), str(output_video_path)):
            cam["output_url"] = f"/outputs/{output_video_path.name}"
        _record_latency("render", time.time() - t3, camera_id)

        cam.update({"status": "completed", "percent": 100, "message": "Done"})
        _log("info", f"{label}: pipeline complete ({time.time() - t0:.1f}s total)", camera_id)
    except Exception as exc:
        logger.exception("Camera job failed: session=%s camera=%s", session_id, camera_id)
        cam.update({"status": "error", "percent": cam.get("percent", 0), "message": str(exc)})
        _log("critical", f"{label}: pipeline failed — {exc}", camera_id)


@app.get("/api/session/{session_id}/camera/{camera_id}/tracks")
def camera_tracks(session_id: str, camera_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    cam = session["cameras"].get(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not cam.get("reid_json_path"):
        raise HTTPException(status_code=409, detail="This camera hasn't finished processing yet")

    from multicam_pipeline import load_track_overlay

    overlay = load_track_overlay(cam["reid_json_path"], cam.get("people", []), cam.get("fps", 25.0))
    return JSONResponse(overlay)


@app.get("/api/logs")
def get_logs(since: int = 0):
    return JSONResponse({"logs": LOGS[since:], "next_since": len(LOGS)})


@app.get("/api/telemetry")
def get_telemetry():
    gpu = {"available": False}
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            gpu = {
                "available": True,
                "name": torch.cuda.get_device_name(idx),
                "vram_used_mb": round(torch.cuda.memory_allocated(idx) / (1024 ** 2), 1),
                "vram_reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024 ** 2), 1),
                "vram_total_mb": round(torch.cuda.get_device_properties(idx).total_memory / (1024 ** 2), 1),
            }
        except Exception:
            gpu = {"available": False}

    active_cameras = []
    for session in SESSIONS.values():
        for cam in session["cameras"].values():
            if cam["status"] == "running":
                active_cameras.append(cam["label"])

    drop_rates = [
        cam["frame_drop_rate"]
        for session in SESSIONS.values()
        for cam in session["cameras"].values()
        if "frame_drop_rate" in cam
    ]
    avg_drop_rate = round(sum(drop_rates) / len(drop_rates), 4) if drop_rates else 0.0

    return JSONResponse(
        {
            "gpu": gpu,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "active_cameras": active_cameras,
            "avg_frame_drop_rate": avg_drop_rate,
            "latency_samples": LATENCY_SAMPLES[-40:],
        }
    )


# ══════════════════════════════════════════════════════════════════════════
#  Watch-list alerts API
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/alerts")
def api_alerts():
    """Return all watch-list alert events (alerts.json), newest first, minus
    the ones the operator marked as handled. Every alert carries the same
    shape the UI expects: person, flag, details, source, camera_id, track_id,
    similarity, crop_url, first_seen_sec, last_seen_sec, time, id."""
    dismissed = _read_dismissed()
    alerts = []
    for ev in _read_alerts():
        alert_id = _alert_event_id(ev)
        if alert_id in dismissed:
            continue
        ev = dict(ev)
        ev["id"] = alert_id
        alerts.append(ev)
    alerts.sort(key=lambda a: a.get("time") or "", reverse=True)
    return JSONResponse({"alerts": alerts})


@app.delete("/api/alerts/{alert_id}")
def api_delete_alert(alert_id: str):
    """Dismiss an alert after it has been handled: it is removed from
    alerts.json entirely (or recorded in dismissed_alerts.json if the event
    is already gone) so it never comes back."""
    alerts = _read_alerts()
    kept = []
    removed = False
    for ev in alerts:
        if _alert_event_id(ev) == alert_id:
            removed = True
            continue
        kept.append(ev)
    if removed:
        ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ALERTS_PATH, "w", encoding="utf-8") as f:
            json.dump(kept[:200], f, indent=2, ensure_ascii=False)
        _log("info", f"Alert dismissed: {alert_id}")
        return JSONResponse({"deleted": True, "alert_id": alert_id})

    # event not (or no longer) in the file — remember it as handled anyway
    dismissed = _read_dismissed()
    dismissed.add(alert_id)
    _write_dismissed(dismissed)
    return JSONResponse({"deleted": True, "alert_id": alert_id})


# ══════════════════════════════════════════════════════════════════════════
#  Legacy single-video quick-test endpoints (unchanged behaviour)
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/process")
def process_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    safe_name = Path(file.filename).name
    job_id = uuid.uuid4().hex[:10]
    input_path = UPLOAD_DIR / f"{job_id}_{safe_name}"

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    detections_path = OUTPUT_DIR / f"{job_id}_detections.json"
    tracking_path = OUTPUT_DIR / f"{job_id}_tracking.json"
    reid_path = OUTPUT_DIR / f"{job_id}_reid.json"
    output_video_path = OUTPUT_DIR / f"{job_id}_reid.mp4"

    JOBS[job_id] = {
        "status": "queued",
        "percent": 0,
        "message": "Queued",
        "output_url": None,
        "output_name": None,
    }

    background_tasks.add_task(
        _run_pipeline_job,
        job_id,
        input_path,
        detections_path,
        tracking_path,
        reid_path,
        output_video_path,
        device,
    )

    return JSONResponse({"job_id": job_id})


@app.get("/api/progress/{job_id}")
def get_progress(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(JOBS[job_id])


def _run_pipeline_job(job_id, input_path, detections_path, tracking_path, reid_path, output_video_path, device):
    try:
        JOBS[job_id].update({"status": "running", "percent": 5, "message": "Starting"})

        JOBS[job_id].update({"percent": 20, "message": "Running detection"})
        run_detection(
            str(input_path),
            str(detections_path),
            conf_threshold=0.6,
            min_height=50,
            min_area_ratio=0.001,
            device=device,
            stride=1,
        )
        _ensure_artifact(detections_path, "Detection output")

        JOBS[job_id].update({"percent": 55, "message": "Running tracking"})
        run_tracking(str(input_path), str(detections_path), str(tracking_path))
        _ensure_artifact(tracking_path, "Tracking output")

        JOBS[job_id].update({"percent": 80, "message": "Running re-identification"})
        run_reid_pipeline(str(input_path), str(tracking_path), str(reid_path), device=device)
        _ensure_artifact(reid_path, "Re-ID output")

        JOBS[job_id].update({"percent": 92, "message": "Rendering output video"})
        if not render_reid_video(str(input_path), str(reid_path), str(output_video_path)):
            JOBS[job_id].update({"status": "error", "message": "Failed to render output video"})
            return
        _ensure_artifact(output_video_path, "Rendered video")

        JOBS[job_id].update(
            {
                "status": "completed",
                "percent": 100,
                "message": "Completed",
                "output_url": f"/outputs/{output_video_path.name}",
                "output_name": output_video_path.name,
            }
        )
    except Exception as exc:
        JOBS[job_id].update({"status": "error", "message": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
