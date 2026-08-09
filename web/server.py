"""
Surveillance Re-ID dashboard API.

Exposes every feature of the project through a small JSON API:

  * System overview / feature status
  * Multi-camera configuration + rendered results
  * Global cross-camera identities (from outputs/cross_camera/global_identities.json)
  * Person registration (multi-image) + identity DB + face DB management
  * Face-based person search across surveillance videos (highlight clips +
    contact sheets)
  * Full multicam pipeline runner (subprocess with live log streaming)

Background work (registration, search, pipeline) runs as jobs; clients poll
GET /api/jobs/{id} for progress + logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Mixing PyTorch (libiomp) and onnxruntime (ArcFace/insightface) inside one
# long-lived process can deadlock when both spawn their OpenMP thread pools.
# Cap the intra-op threads for this process (registration / face search are
# light) and keep the pipeline subprocess exempted below so full runs keep
# their parallelism.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("web.server")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

WEB_DIR = ROOT_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
UPLOAD_DIR = WEB_DIR / "uploads"
OUTPUT_DIR = WEB_DIR / "outputs"
OUTPUTS_ROOT = ROOT_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = WEB_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Surveillance Re-ID Dashboard")

# ── static mounts ─────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
# Everything the pipeline produces lives under the project outputs/ folder;
# serve it read-only so the UI can play rendered videos, crop galleries,
# registration photos, face-search clips and contact sheets.
app.mount("/data/outputs", StaticFiles(directory=OUTPUTS_ROOT), name="data-outputs")
# Raw camera input videos, so the UI can show the unprocessed source footage
# too (before the pipeline has rendered an annotated output).
INPUT_DIR = ROOT_DIR / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/data/input", StaticFiles(directory=INPUT_DIR), name="data-input")
# Uploaded videos that became system cameras (Video Processing page) are served
# here so the Overview can also show their raw source footage.
app.mount("/data/uploads", StaticFiles(directory=UPLOAD_DIR), name="data-uploads")


# ── tiny background job store ─────────────────────────────────────────────────
JOBS: Dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _new_job(kind: str) -> str:
    job_id = uuid.uuid4().hex[:10]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "percent": 0,
            "message": "Queued",
            "log": [],
            "result": None,
            "error": None,
        }
    return job_id


def _append_log(job_id: str, msg: str, percent: Optional[int] = None):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["log"].append(msg)
        if len(job["log"]) > 800:
            job["log"] = job["log"][-800:]
        if percent is not None:
            job["percent"] = int(max(0, min(100, percent)))
            job["message"] = msg


def _finish_job(job_id: str, error: str = None, result=None):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        if error:
            job["status"] = "error"
            job["error"] = error
            job["message"] = f"Error: {error}"
        else:
            job["status"] = "done"
            job["percent"] = 100
            job["message"] = "Completed"
            job["result"] = result


def _run_in_thread(fn):
    threading.Thread(target=fn, daemon=True).start()


# ── job queues ──────────────────────────────────────────────────────────────
# Video jobs (uploaded / live clips) run in a bounded thread pool so several
# videos can be processed at once ("as fast as possible"), each in its own
# subprocess. The multi-camera pipeline stays on its own single worker so two
# heavy cross-camera runs never fight over the GPU / model files at once.
VIDEO_WORKERS = max(1, min(3, (os.cpu_count() or 2) // 2))
_VIDEO_EXECUTOR = ThreadPoolExecutor(max_workers=VIDEO_WORKERS, thread_name_prefix="video")
_PIPE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")


def _enqueue(job_id: str, run_fn, pool: str = "video"):
    """Run `run_fn(job_id)` on the matching worker pool."""
    executor = _VIDEO_EXECUTOR if pool == "video" else _PIPE_EXECUTOR

    def _guard():
        try:
            run_fn(job_id)
        except Exception as e:
            logger.exception(f"Job {job_id} crashed")
            _finish_job(job_id, error=f"Worker crashed: {e}")

    executor.submit(_guard)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _db_fingerprint() -> str:
    """Hash the registered-person database so processed-video caches are
    invalidated the moment the DB is changed / edited / rebuilt. Covers the
    identity DB, the face DB and every registered photo."""
    h = hashlib.sha256()
    reg = OUTPUTS_ROOT / "registration"
    roots = [reg / "identity_db.json", reg / "face_db.json", reg / "images"]
    for root in roots:
        if not root.exists():
            continue
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if not p.is_file():
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                h.update(p.relative_to(reg).as_posix().encode("utf-8", "replace"))
                h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
        else:
            try:
                st = root.stat()
            except OSError:
                continue
            h.update(root.name.encode())
            h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()


def _video_cache_key(db_fp: str, file_hash: str) -> Path:
    return CACHE_DIR / f"{db_fp[:12]}_{file_hash}.json"


def _load_video_cache(db_fp: str, file_hash: str):
    path = _video_cache_key(db_fp, file_hash)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_video_cache(db_fp: str, file_hash: str, result: dict):
    if not file_hash:
        return
    try:
        _video_cache_key(db_fp, file_hash).write_text(
            json.dumps(result), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"Could not write video cache: {e}")


def _cached_output_ok(result: dict) -> bool:
    """A cache entry is usable only when its rendered video still exists."""
    out_name = (result or {}).get("output_name")
    if not out_name:
        return False
    p = OUTPUT_DIR / out_name
    return p.exists() and p.stat().st_size > 0


def _purge_legacy_caches():
    """Drop pre-DB-fingerprint cache entries (bare sha256 names) that can never
    be matched by the current key scheme and would silently force re-processing
    forever otherwise."""
    try:
        for p in CACHE_DIR.glob("*.json"):
            if "_" not in p.stem:
                p.unlink(missing_ok=True)
    except Exception:
        pass


_purge_legacy_caches()


def _data_url(rel: Path) -> str:
    """Convert a path under outputs/ into a web-accessible URL."""
    rel = rel.resolve()
    root = OUTPUTS_ROOT.resolve()
    try:
        r = rel.relative_to(root)
    except ValueError:
        return ""
    return "/data/outputs/" + r.as_posix()


# ── shared helpers ────────────────────────────────────────────────────────────
def _load_json(rel: str):
    path = ROOT_DIR / rel
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return None


def _rendered_video_for(cam_id: str) -> Optional[Path]:
    """Locate a camera's rendered output (mp4 when ffmpeg re-encoded it, webm
    otherwise) so the dashboard serves whatever actually exists and plays."""
    rendered_dir = OUTPUTS_ROOT / "rendered"
    for name in (f"{cam_id}_results.mp4", f"{cam_id}_results.webm"):
        p = rendered_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


# ── camera identity registry ──────────────────────────────────────────────────
# Videos processed on the Video Processing page become the system's cameras
# (cam1, cam2, ... in upload order). Each camera keeps a copy of its rendered
# output at outputs/rendered/{cam_id}_results.* and a record of its source
# video, so the Overview cameras card and rendered list reflect uploads instead
# of a hard-coded config.
CAMERA_SOURCES_PATH = OUTPUTS_ROOT / "rendered" / "camera_sources.json"
_VIDEO_EXT = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


def _cam_sort_key(cam_id: str):
    digits = "".join(ch for ch in cam_id if ch.isdigit())
    return (int(digits) if digits else 10 ** 9, cam_id)


def _load_camera_sources() -> dict:
    try:
        if CAMERA_SOURCES_PATH.exists():
            data = json.loads(CAMERA_SOURCES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception as e:
        logger.warning(f"Could not read camera sources: {e}")
    return {}


def _save_camera_source(camera_id: str, source: str):
    sources = _load_camera_sources()
    sources[camera_id] = source
    CAMERA_SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        CAMERA_SOURCES_PATH.write_text(
            json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"Could not write camera sources: {e}")


def _publish_camera(camera_id: str, result: dict, input_path: Path):
    """Give a processed video a stable camera identity: copy the worker's
    content-addressed rendered output to outputs/rendered/{cam_id}_results.* and
    remember the source video so the Overview + rendered list pick this camera up."""
    if not camera_id:
        return
    out_name = (result or {}).get("output_name")
    if not out_name:
        return
    src = OUTPUT_DIR / out_name
    if not src.exists() or src.stat().st_size == 0:
        return
    dst = OUTPUTS_ROOT / "rendered" / f"{camera_id}_results{src.suffix}"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        for other in dst.parent.glob(f"{camera_id}_results.*"):
            if other.suffix.lower() != src.suffix.lower():
                other.unlink(missing_ok=True)
        if dst.exists():
            dst.unlink(missing_ok=True)
        shutil.copy2(src, dst)
        try:
            source = str(input_path.resolve().relative_to(ROOT_DIR)).replace("\\", "/")
        except Exception:
            source = str(input_path.resolve()).replace("\\", "/")
        _save_camera_source(camera_id, source)
    except Exception as e:
        logger.warning(f"Could not publish camera {camera_id}: {e}")


def _cameras_with_status():
    """Cameras = the pipeline's configured CAMERAS (cam1/cam2/cam3), always
    shown on the dashboard. Each camera is enriched with:
      * the rendered output (outputs/rendered/{cam_id}_results.*) when a video
        was processed on the Video Processing page,
      * the raw source video URL (input/ or web/uploads/) when it exists, and
      * per-camera stats from the last multi-camera run
        (outputs/cross_camera/global_identities.json).
    Any extra camera that only exists as a rendered output is appended, so
    nothing produced by the Video Processing page disappears."""
    sources = _load_camera_sources()
    try:
        from multicamera.camera_config import CAMERAS
    except Exception:
        CAMERAS = []
    config_sources = {c["camera_id"]: c["source"] for c in CAMERAS}

    gid_data = _load_json("outputs/cross_camera/global_identities.json") or {}
    frames = {k: v for k, v in gid_data.items() if k != "global_identities"}

    def _cam_stats(cam_id: str):
        cam_frames = frames.get(cam_id) or {}
        person_frames = sum(1 for people in cam_frames.values() if people)
        identified = sorted(
            {
                p["name"]
                for people in cam_frames.values()
                for p in people
                if p.get("name")
            }
        )
        return person_frames, identified

    rendered_dir = OUTPUTS_ROOT / "rendered"
    rendered_ids = set()
    if rendered_dir.exists():
        for v in rendered_dir.iterdir():
            if v.suffix.lower() not in _VIDEO_EXT or not v.stem.endswith("_results"):
                continue
            rendered_ids.add(v.stem[: -len("_results")])

    ordered = [c["camera_id"] for c in CAMERAS]
    for cam_id in sorted(rendered_ids, key=_cam_sort_key):
        if cam_id not in ordered:
            ordered.append(cam_id)

    out = []
    for cam_id in ordered:
        source = sources.get(cam_id) or config_sources.get(cam_id, "")
        src_path = Path(source) if source else None
        exists = False
        if src_path is not None:
            if not src_path.is_absolute():
                src_path = ROOT_DIR / src_path
            exists = src_path.exists()
        rendered = _rendered_video_for(cam_id)
        preview = rendered_dir / f"{cam_id}.png"
        input_url = None
        if exists and src_path:
            try:
                if INPUT_DIR in src_path.parents:
                    input_url = "/data/input/" + src_path.relative_to(INPUT_DIR).as_posix()
                elif UPLOAD_DIR in src_path.parents:
                    input_url = "/data/uploads/" + src_path.relative_to(UPLOAD_DIR).as_posix()
            except Exception:
                input_url = None
        person_frames, identified = _cam_stats(cam_id)
        rendered_url = _data_url(rendered) if rendered is not None else None
        out.append(
            {
                "camera_id": cam_id,
                "source": source,
                "source_name": src_path.name if src_path else "",
                "source_exists": exists,
                "rendered": rendered is not None,
                "rendered_url": rendered_url,
                "input_url": input_url,
                "play_url": rendered_url or input_url,
                "preview_url": _data_url(preview) if preview.exists() else None,
                "people_frames": person_frames,
                "identified": identified,
            }
        )
    return out


def _face_db():
    from registration.face_db import FaceIdentityDB

    return FaceIdentityDB()


def _identity_db():
    from registration.identity_db import IdentityDatabase

    return IdentityDatabase()


# ── alerts (watch-list sightings) ────────────────────────────────────────────
ALERTS_PATH = OUTPUTS_ROOT / "alerts.json"


def _read_alerts():
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
    alerts = _read_alerts()
    alert.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    alerts.insert(0, alert)
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump(alerts[:200], f, indent=2, ensure_ascii=False)


# Alerts that were "handled" / dismissed from the dashboard. Persistent events
# are removed from alerts.json entirely; live multi-camera alerts are keyed by
# (person, global_id) so they stay dismissed until that person is seen again
# with a different identity.
DISMISSED_PATH = OUTPUTS_ROOT / "dismissed_alerts.json"


def _read_dismissed() -> dict:
    try:
        if DISMISSED_PATH.exists():
            data = json.loads(DISMISSED_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("events", [])
                data.setdefault("multicam", [])
                return data
    except Exception as e:
        logger.warning(f"Could not read dismissed alerts: {e}")
    return {"events": [], "multicam": []}


def _write_dismissed(dismissed: dict):
    DISMISSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        DISMISSED_PATH.write_text(
            json.dumps(dismissed, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"Could not write dismissed alerts: {e}")


def _alert_event_id(ev: dict) -> str:
    """Stable id for a persisted alert event (alerts.json)."""
    key = {
        "person": ev.get("person"),
        "source": ev.get("source"),
        "time": ev.get("time"),
        "camera_id": ev.get("camera_id"),
        "camera_ids": sorted(ev.get("cameras") or []),
        "track_id": ev.get("track_id"),
        "frames": ev.get("frames"),
    }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_alert_event(ev: dict) -> dict:
    ev = dict(ev)
    ev.setdefault("cameras", [ev["camera_id"]] if ev.get("camera_id") else [])
    return ev


def _flag_label(flag):
    from registration.identity_db import FLAGS
    return flag if flag in FLAGS else "normal"


# ── single-video job runner (subprocess) ─────────────────────────────────────
def _video_percent(line: str):
    """Map single-video worker progress lines to a job percentage."""
    if "Recording" in line:
        return 10
    if "STEP 1" in line or "detecting" in line.lower():
        return 20
    if "STEP 2" in line or "tracking" in line.lower():
        return 45
    if "STEP 3" in line or "re-identif" in line.lower() or "reid" in line.lower():
        return 70
    if "STEP 4" in line or "matching" in line.lower() or "identified" in line.lower():
        return 85
    if "STEP 5" in line or "render" in line.lower():
        return 95
    return None


def _run_video_subprocess(job_id, input_path, label, device, video_hash="", db_fp="", camera_id="cam"):
    """Run detect -> track -> re-ID -> identify -> render in a SEPARATE process
    (web/video_worker.py) so the dashboard stays responsive. Progress lines are
    streamed into the job console; the worker writes the result JSON."""
    args = [
        sys.executable, "-u",
        str(WEB_DIR / "single_video_job.py"),
        job_id, str(input_path), label, device,
    ]
    if video_hash:
        args.append(video_hash)
    if camera_id:
        args.append(camera_id)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("OMP_NUM_THREADS", None)
    env.pop("KMP_DUPLICATE_LIB_OK", None)

    _append_log(job_id, f"Starting worker for '{label}'...", 3)
    proc = subprocess.Popen(
        args, cwd=str(ROOT_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("JOB_ERROR"):
            _finish_job(job_id, error=line.replace("JOB_ERROR", "").strip())
            proc.wait()
            return
        _append_log(job_id, line, _video_percent(line))
    code = proc.wait()

    result_path = OUTPUT_DIR / f"{job_id}_result.json"
    if code == 0 and result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if video_hash and db_fp:
                _write_video_cache(db_fp, video_hash, result)
            _publish_camera(camera_id, result, input_path)
            _append_log(job_id, f"Complete: {len(result.get('persons', []))} registered person(s) "
                                f"identified · {result.get('alert_count', 0)} alert(s).", 100)
            _finish_job(job_id, result=result)
        except Exception as e:
            _finish_job(job_id, error=f"Could not read job result: {e}")
    else:
        _finish_job(job_id, error=f"Worker exited with code {code}")


def _identity_crops(gid, frames, limit=12):
    """Find saved person crops for a global identity across cameras."""
    from multicamera.camera_config import CAMERAS

    crops_dir = OUTPUTS_ROOT / "multicam" / "crops"
    candidates = []
    for cam in CAMERAS:
        cam_id = cam["camera_id"]
        for frame_key, people in (frames.get(cam_id) or {}).items():
            for p in people:
                if str(p.get("global_id")) != str(gid):
                    continue
                crop = crops_dir / cam_id / f"track{p['id']}_frame{frame_key}.jpg"
                if crop.exists():
                    candidates.append(
                        {
                            "camera_id": cam_id,
                            "frame": int(frame_key),
                            "track_id": int(p["id"]),
                            "url": _data_url(crop),
                        }
                    )
    # even sampling if we have more than `limit`
    if len(candidates) > limit:
        step = len(candidates) / limit
        candidates = [candidates[int(i * step)] for i in range(limit)]
    return candidates


# ── pages ─────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


# ── overview ──────────────────────────────────────────────────────────────────
@app.get("/api/overview")
def api_overview():
    cameras = _cameras_with_status()
    gid_data = _load_json("outputs/cross_camera/global_identities.json") or {}
    identity_map = gid_data.get("global_identities", {})
    rendered_dir = OUTPUTS_ROOT / "rendered"
    face_index_files = sorted((OUTPUTS_ROOT / "face_search").glob("*_faces.json")) \
        if (OUTPUTS_ROOT / "face_search").exists() else []
    try:
        face_db = _face_db()
        persons = sorted(face_db.list_persons())
        face_db_path = OUTPUTS_ROOT / "registration" / "face_db.json"
        has_face_db = face_db_path.exists()
    except Exception as e:
        persons = []
        has_face_db = False
        logger.warning(f"Face DB unavailable in overview: {e}")

    try:
        id_db = _identity_db()
    except Exception:
        id_db = None

    identities = []
    for gid, meta in sorted(identity_map.items(), key=lambda kv: int(kv[0])):
        name = meta.get("name")
        flag = "normal"
        if name and id_db is not None:
            try:
                flag = _flag_label((id_db.get_person(name) or {}).get("metadata", {}).get("flag", "normal"))
            except Exception:
                flag = "normal"
        identities.append(
            {
                "global_id": int(gid),
                "name": name,
                "flag": flag,
                "cameras": meta.get("cameras_seen_on", []),
                "observations": meta.get("observations", 0),
                "similarity": round(meta.get("name_similarity") or 0.0, 4),
                "name_source": meta.get("name_source", ""),
            }
        )

    return JSONResponse(
        {
            "cameras": cameras,
            "rendered_count": len(list(rendered_dir.glob("*_results.mp4"))) + len(list(rendered_dir.glob("*_results.webm"))),
            "face_index_count": len(face_index_files),
            "face_indexes": [p.stem.replace("_faces", "") for p in face_index_files],
            "registered_persons": persons,
            "has_face_db": has_face_db,
            "global_identities_count": len(identity_map),
            "global_named_count": sum(1 for v in identity_map.values() if v.get("name")),
            "identities": identities,
            "models": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "torch": torch.__version__,
            },
        }
    )


# ── cameras / rendered ────────────────────────────────────────────────────────
@app.get("/api/cameras")
def api_cameras():
    return JSONResponse({"cameras": _cameras_with_status()})


@app.get("/api/rendered")
def api_rendered():
    rendered_dir = OUTPUTS_ROOT / "rendered"
    videos = []
    if rendered_dir.exists():
        for v in sorted(list(rendered_dir.glob("*_results.mp4")) + list(rendered_dir.glob("*_results.webm"))):
            preview = v.with_suffix(".png")
            videos.append(
                {
                    "name": v.name,
                    "url": _data_url(v),
                    "preview_url": _data_url(preview) if preview.exists() else None,
                }
            )
    return JSONResponse({"rendered": videos})


# ── global identities ─────────────────────────────────────────────────────────
@app.get("/api/global-identities")
def api_global_identities():
    data = _load_json("outputs/cross_camera/global_identities.json") or {}
    frames = {k: v for k, v in data.items() if k != "global_identities"}
    identity_map = data.get("global_identities", {})

    identities = []
    for gid, meta in sorted(identity_map.items(), key=lambda kv: int(kv[0])):
        cameras_seen_on = meta.get("cameras_seen_on", [])
        observations = {}
        for cam in cameras_seen_on:
            count = sum(
                1 for people in (frames.get(cam) or {}).values()
                if any(str(p.get("global_id")) == str(gid) for p in people)
            )
            observations[cam] = count
        name = None
        for cam in cameras_seen_on:
            for people in (frames.get(cam) or {}).values():
                for p in people:
                    if str(p.get("global_id")) == str(gid) and p.get("name"):
                        name = p["name"]
                        break
                if name:
                    break
            if name:
                break
        identities.append(
            {
                "global_id": int(gid),
                "cameras_seen_on": cameras_seen_on,
                "observations": observations,
                "total_observations": sum(observations.values()),
                "name": name,
                "crop": (_identity_crops(gid, frames, limit=1) or [{}])[0].get("url"),
            }
        )

    return JSONResponse(
        {
            "identities": identities,
            "frame_counts": {
                cam: sum(1 for people in (frames.get(cam) or {}).values() if people)
                for cam in ("cam1", "cam2", "cam3", "cam4")
            },
            "frames": frames,
        }
    )


@app.get("/api/identity/{gid}/crops")
def api_identity_crops(gid: int):
    data = _load_json("outputs/cross_camera/global_identities.json") or {}
    frames = {k: v for k, v in data.items() if k != "global_identities"}
    return JSONResponse({"global_id": int(gid), "crops": _identity_crops(gid, frames, limit=24)})


# ── persons / registration ────────────────────────────────────────────────────
@app.get("/api/persons")
def api_persons():
    persons = []
    face_db = _face_db()
    id_db = _identity_db()
    for name in sorted(set(face_db.list_persons()) | set(id_db.list_persons())):
        images_dir = OUTPUTS_ROOT / "registration" / "images" / name
        images = sorted(images_dir.glob("*")) if images_dir.exists() else []
        thumb = _data_url(images[0]) if images else None
        face_rec = face_db.get_person(name)
        id_rec = id_db.get_person(name)
        meta = (id_rec or {}).get("metadata", {})
        persons.append(
            {
                "name": name,
                "num_images": len(images),
                "num_faces": (face_rec or {}).get("num_faces", 0),
                "images": [_data_url(p) for p in images],
                "thumbnail": thumb,
                "registered_at": meta.get("registered_at"),
                "searchable": bool((face_rec or {}).get("face_encodings")),
                "person_id": meta.get("person_id"),
                "flag": _flag_label(meta.get("flag", "normal")),
                "details": meta.get("details", ""),
            }
        )
    return JSONResponse({"persons": persons})


@app.patch("/api/persons/{name}")
def api_update_person(name: str, payload: dict):
    """Update a registered person's profile fields (person_id / flag / details)."""
    id_db = _identity_db()
    if id_db.get_person(name) is None:
        raise HTTPException(status_code=404, detail=f"'{name}' is not registered")
    try:
        id_db.update_metadata(name, **{
            k: v for k, v in payload.items() if k in ("person_id", "flag", "details") and v is not None
        })
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"updated": name, "person": id_db.get_person(name)})


@app.post("/api/persons")
async def api_register_person(name: str = Form(...), files: list[UploadFile] = File(...),
                              person_id: str = Form(""), flag: str = Form("normal"),
                              details: str = Form("")):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Person name cannot be empty")
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one photo")
    flag = _flag_label(flag)

    job_id = _new_job("register")
    saved = []
    for i, f in enumerate(files):
        if not f.filename:
            continue
        tmp = UPLOAD_DIR / f"{job_id}_{i:02d}_{Path(f.filename).name}"
        with tmp.open("wb") as buf:
            shutil.copyfileobj(f.file, buf)
        saved.append(str(tmp))
    if not saved:
        raise HTTPException(status_code=400, detail="No readable files uploaded")

    def _run():
        try:
            _append_log(job_id, f"Registering '{name.strip()}' with {len(saved)} photo(s)...", 10)
            from registration.register_person import register_person
            record = register_person(name.strip(), saved,
                                     person_id=person_id or None,
                                     flag=flag, details=details.strip())
            _append_log(job_id, f"Body embedding registered ({record['metadata']['num_images']} image(s)).", 60)
            _append_log(job_id, "Rebuilding face DB for face-search...", 75)
            from registration.face_db import build_from_registration
            added = build_from_registration(_face_db())
            num_faces = next((n for nm, n in added if nm == name.strip()), 0)
            _append_log(job_id, f"Face DB ready: '{name.strip()}' has {num_faces} face encoding(s).", 100)
            _finish_job(job_id, result={"name": name.strip(), "num_images": record["metadata"]["num_images"], "num_faces": num_faces})
        except Exception as e:
            logger.exception("Registration failed")
            _finish_job(job_id, error=str(e))
        finally:
            for p in saved:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass

    _run_in_thread(_run)
    return JSONResponse({"job_id": job_id})


@app.delete("/api/persons/{name}")
def api_delete_person(name: str):
    id_db = _identity_db()
    removed = id_db.delete_person(name)
    images_dir = OUTPUTS_ROOT / "registration" / "images" / name
    if images_dir.exists():
        shutil.rmtree(images_dir, ignore_errors=True)
        removed = True
    if removed:
        from registration.face_db import build_from_registration
        build_from_registration(_face_db())
    return JSONResponse({"deleted": name, "removed": removed})


@app.post("/api/persons/rebuild-face-db")
def api_rebuild_face_db():
    job_id = _new_job("rebuild-face-db")

    def _run():
        try:
            _append_log(job_id, "Scanning registered photos...", 10)
            from registration.face_db import build_from_registration
            added = build_from_registration(_face_db())
            for name, n in added:
                _append_log(job_id, f"  '{name}': {n} face encoding(s)", None)
            _append_log(job_id, f"Rebuilt face DB for {len(added)} person(s).", 100)
            _finish_job(job_id, result={"persons": len(added)})
        except Exception as e:
            logger.exception("Face DB rebuild failed")
            _finish_job(job_id, error=str(e))

    _run_in_thread(_run)
    return JSONResponse({"job_id": job_id})


# ── face search ───────────────────────────────────────────────────────────────
@app.get("/api/search/persons")
def api_searchable_persons():
    face_db = _face_db()
    out = []
    for name in sorted(face_db.list_persons()):
        rec = face_db.get_person(name)
        out.append({"name": name, "num_faces": rec.get("num_faces", 0)})
    return JSONResponse({"persons": out})


@app.get("/api/search/results/{name}")
def api_search_results(name: str):
    path = OUTPUTS_ROOT / "face_search" / f"{name}_results.json"
    if not path.exists():
        return JSONResponse({"exists": False, "name": name})
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"exists": True, "name": name, "report": data})


@app.post("/api/search")
def api_search(payload: dict):
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    render = bool(payload.get("render", True))
    min_frames = int(payload.get("min_frames") or 3)
    threshold = payload.get("threshold")
    if threshold is not None:
        threshold = float(threshold)
    videos = payload.get("videos") or []

    job_id = _new_job("search")
    from reidentification.face_encoder import confirm_distance
    if threshold is None:
        threshold = confirm_distance()

    def _run():
        from reidentification.face_search import (
            FACE_INDEX_DIR, get_index, search_video, group_matches,
            render_highlight_video, build_contact_sheet,
        )
        from registration.face_db import MIN_SEGMENT_FRAMES, build_from_registration
        nonlocal videos

        try:
            _append_log(job_id, f"Searching '{name}' (face distance threshold {threshold:.3f})...", 5)
            face_db = _face_db()
            if name not in face_db.list_persons():
                _append_log(job_id, "Face DB missing entry - rebuilding from registered photos...", 10)
                build_from_registration(face_db)
            rec = face_db.get_person(name)
            if not rec or not rec["face_encodings"]:
                raise RuntimeError(
                    f"'{name}' has no usable face in the face DB. "
                    "Register face-visible photos first."
                )
            encodings = rec["face_encodings"]
            query_images = []
            import cv2
            for p in rec["image_paths"]:
                img = cv2.imread(str(p))
                if img is not None:
                    query_images.append(img)
            _append_log(job_id, f"Query ready: {len(encodings)} face encoding(s).", 15)

            if not videos:
                videos = [str(ROOT_DIR / "input" / f"video{i}.mp4") for i in range(1, 5)]
            videos = [v for v in videos if Path(v).exists()]
            found_videos = []

            total = len(videos)
            for idx, vp in enumerate(videos):
                base = Path(vp).stem
                _append_log(job_id, f"[{base}] loading face index...", 15 + int(60 * idx / max(1, total)))
                index_data = get_index(vp, f"outputs/{base}_detections.json")
                if index_data is None:
                    _append_log(job_id, f"[{base}] no face index / detections, skipping.")
                    continue
                matches = search_video(index_data, encodings, distance_threshold=threshold)
                if not matches:
                    _append_log(job_id, f"[{base}] no matches.")
                    continue
                segments = group_matches(matches, index_data["fps"])
                segments = [s for s in segments if s["n_frames"] >= min_frames]
                if not segments:
                    _append_log(job_id, f"[{base}] {len(matches)} frame(s) matched but none form a confirmed segment.")
                    continue
                for seg in segments:
                    _append_log(job_id, f"[{base}] MATCH frames {seg['first_frame']}-{seg['last_frame']} "
                                        f"({seg['first_time']} to {seg['last_time']}) "
                                        f"sim={seg['best_similarity']:.2f}, {seg['n_frames']} frame(s)")
                video_entry = {
                    "video": base,
                    "fps": index_data["fps"],
                    "segments": segments,
                }
                if render:
                    out_vid = OUTPUTS_ROOT / "face_search" / f"{name}_{base}.mp4"
                    out_sheet = OUTPUTS_ROOT / "face_search" / f"{name}_{base}_contact.jpg"
                    _append_log(job_id, f"[{base}] rendering highlight clip + contact sheet...")
                    render_highlight_video(vp, index_data, name, segments, out_vid)
                    build_contact_sheet(query_images, index_data, segments, vp, out_sheet,
                                        name=name, matches=matches)
                    video_entry["highlight_url"] = _data_url(out_vid) if out_vid.exists() else None
                    video_entry["contact_url"] = _data_url(out_sheet) if out_sheet.exists() else None
                found_videos.append(video_entry)

            report = {
                "name": name,
                "threshold": threshold,
                "min_frames": min_frames,
                "videos_searched": [Path(v).stem for v in videos],
                "found": len(found_videos) > 0,
                "matches": found_videos,
            }
            out_path = OUTPUTS_ROOT / "face_search" / f"{name}_results.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            if found_videos:
                _append_log(job_id, f"'{name}' FOUND in {len(found_videos)} video(s). Report saved.", 100)
            else:
                _append_log(job_id, f"'{name}' NOT found in any searched video.", 100)
            _finish_job(job_id, result=report)
        except Exception as e:
            logger.exception("Search failed")
            _finish_job(job_id, error=str(e))

    _run_in_thread(_run)
    return JSONResponse({"job_id": job_id})


# ── pipeline runner ───────────────────────────────────────────────────────────
@app.post("/api/pipeline/run")
def api_pipeline_run(payload: dict):
    job_id = _new_job("pipeline")
    args = [
        sys.executable, "-u", str(ROOT_DIR / "run_integrated_pipeline.py"),
        "--device", payload.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        "--conf-threshold", str(payload.get("conf_threshold", 0.35)),
        "--cross-cam-threshold", str(payload.get("cross_cam_threshold", 0.60)),
    ]
    if payload.get("max_frames"):
        args += ["--max-frames", str(int(payload["max_frames"]))]
    if payload.get("wanted_only"):
        args.append("--wanted-only")
    else:
        args.append("--track-everyone")
    if payload.get("track_level"):
        args.append("--track-level")
    if payload.get("no_cache"):
        args.append("--no-cache")
    if payload.get("skip_names"):
        args.append("--skip-names")
    if payload.get("no_face_names"):
        args.append("--no-face-names")
    if payload.get("no_body_gate"):
        args.append("--no-body-gate")
    if payload.get("debug_trace"):
        args.append("--debug-trace")
    if payload.get("output"):
        args += ["--output", str(payload["output"])]

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # The pipeline is a self-contained CLI process (like running
    # run_integrated_pipeline.py directly) — restore full thread parallelism.
    env.pop("OMP_NUM_THREADS", None)
    env.pop("KMP_DUPLICATE_LIB_OK", None)

    def _run():
        _append_log(job_id, "Starting integrated multicam pipeline...", 2)
        _append_log(job_id, "Command: " + " ".join(args), None)
        proc = subprocess.Popen(
            args, cwd=str(ROOT_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            _append_log(job_id, line, None)
            percent = _parse_pipeline_percent(line)
            if percent is not None:
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                    if job:
                        job["percent"] = percent
                        job["message"] = line
        code = proc.wait()
        if code == 0:
            _append_log(job_id, "Pipeline finished successfully.", 100)
            _finish_job(job_id, result={"returncode": code})
        else:
            _finish_job(job_id, error=f"Pipeline exited with code {code}")

    _enqueue(job_id, lambda _jid: _run(), pool="pipeline")
    return JSONResponse({"job_id": job_id})


def _parse_pipeline_percent(line: str):
    """Best-effort progress estimate from known pipeline stage markers."""
    if "STEP 1" in line or "Detecting" in line or "detection" in line.lower():
        return 25
    if "STEP 2" in line or "tracking" in line.lower():
        return 45
    if "STEP 3" in line or "re-identif" in line.lower() or "reid" in line.lower():
        return 65
    if "cross-camera" in line.lower() or "global identit" in line.lower() or "clustering" in line.lower():
        return 85
    if "rendering" in line.lower() or "render" in line.lower():
        return 95
    return None


# ── recorded-video upload / processing ───────────────────────────────────────
_ALLOWED_VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


@app.post("/api/videos/process")
async def api_process_videos(files: list[UploadFile] = File(...)):
    """Upload one or more recorded videos and run the full detect->track->re-ID
    ->identify->render pipeline on each. Returns a list of job ids - poll
    GET /api/jobs/{id} per video. Identical files (same bytes + same registered
    DB) are served from cache instead of being reprocessed."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    db_fp = _db_fingerprint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jobs = []
    for idx, file in enumerate(files):
        if not file.filename:
            continue
        if not (file.filename or "").lower().endswith(_ALLOWED_VIDEO_EXT):
            raise HTTPException(status_code=400, detail=f"Please upload a video file (.mp4, .avi, .mov, .mkv, .webm): {file.filename}")

        camera_id = f"cam{idx + 1}"
        job_id = _new_job("video")
        safe_name = Path(file.filename).name
        input_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        jobs.append({"job_id": job_id, "filename": safe_name, "camera_id": camera_id})

        def _run(jid: str = job_id, path: Path = input_path, name: str = safe_name, cam: str = camera_id):
            _append_log(jid, f"Received upload '{name}' ({path.stat().st_size / 1e6:.1f} MB) as {cam}.", 2)
            _append_log(jid, "Checking for a previously processed copy of this video...", 4)
            video_hash = _sha256(path)
            cached = _load_video_cache(db_fp, video_hash)
            if cached is not None and _cached_output_ok(cached):
                cached = dict(cached)
                cached["label"] = name
                cached["camera_id"] = cam
                _publish_camera(cam, cached, path)
                _append_log(jid, "This exact video was processed before (same registered DB) - loading the cached result.", 100)
                _append_log(jid, f"Cached: {len(cached.get('persons', []))} registered person(s) "
                                 f"identified · {cached.get('alert_count', 0)} alert(s).", 100)
                _finish_job(jid, result=cached)
                return
            if cached is not None:
                _append_log(jid, "Previous result discarded (rendered output was cleaned) - re-processing.", 5)
            _append_log(jid, "No cached result - queueing for processing.", 5)
            _enqueue(jid, lambda jid_: _run_video_subprocess(jid_, path, name, device, video_hash, db_fp, cam))

        _run_in_thread(_run)

    return JSONResponse({"jobs": jobs})


@app.post("/api/video/process")
async def api_process_video(file: UploadFile = File(...)):
    """Single-video convenience wrapper around /api/videos/process."""
    return await api_process_videos(files=[file])


# ── live webcam (record-then-process) ────────────────────────────────────────
@app.post("/api/live")
async def api_live(payload: dict):
    """Record a short clip from the webcam, then run the same pipeline and
    identify any registered person on screen."""
    import cv2

    duration = int(payload.get("duration", 10))
    duration = max(3, min(duration, 30))
    label = "Live cam"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    job_id = _new_job("live")
    out_path = UPLOAD_DIR / f"{job_id}_live.mp4"

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="No webcam available - plug in a camera or use the recorded-video upload instead.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="Could not initialise video writer for webcam recording")

    def _run():
        try:
            _append_log(job_id, f"Recording {duration}s from webcam...", 5)
            total = int(duration * fps)
            written = 0
            for _ in range(total):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
                if written % max(1, int(fps)) == 0:
                    _append_log(job_id, f"  recorded {written / fps:.0f}s / {duration}s", None)
            cap.release()
            writer.release()
            if written < fps * 0.5:
                _finish_job(job_id, error="Webcam produced no usable frames - check the camera.")
                return
            _append_log(job_id, f"Recording complete ({written / fps:.1f}s). Queuing pipeline worker...", 20)
            _enqueue(job_id, lambda jid: _run_video_subprocess(jid, out_path, label, device, camera_id=""))
        except Exception as e:
            logger.exception("Live job failed")
            _finish_job(job_id, error=str(e))

    _run_in_thread(_run)
    return JSONResponse({"job_id": job_id})


# ── alerts ───────────────────────────────────────────────────────────────────
@app.get("/api/alerts")
def api_alerts():
    """Merge two sources into one normalised list:
      1. flagged registered persons (criminal / missing / person of interest)
         actually recognised in the last multi-camera run
         (outputs/cross_camera/global_identities.json), with per-camera frame
         counts and a crop from the pipeline's crop gallery;
      2. alert events persisted by single-video / live-cam jobs
         (outputs/alerts.json).
    Every alert carries the same shape the UI expects: person, flag, details,
    cameras (list), cameras_frames, frames, similarity, crop_url, source,
    camera_id, gid, time."""
    from multicamera.camera_config import CAMERAS

    id_db = _identity_db()
    flagged = {}
    for name in id_db.list_persons():
        meta = (id_db.get_person(name) or {}).get("metadata", {})
        flag = _flag_label(meta.get("flag", "normal"))
        if flag != "normal":
            flagged[name] = {"flag": flag, "details": meta.get("details", "")}

    alerts = []
    gid_path = OUTPUTS_ROOT / "cross_camera" / "global_identities.json"
    gid_data = _load_json("outputs/cross_camera/global_identities.json") or {}
    frames = {k: v for k, v in gid_data.items() if k != "global_identities"}
    identity_map = gid_data.get("global_identities", {})
    run_time = None
    try:
        if gid_path.exists():
            run_time = datetime.fromtimestamp(gid_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        run_time = None

    crops_dir = OUTPUTS_ROOT / "multicam" / "crops"
    dismissed = _read_dismissed()
    dismissed_multicam = {(str(p), str(g)) for p, g in dismissed.get("multicam", [])}
    dismissed_events = set(dismissed.get("events", []))
    for name, info in flagged.items():
        for gid, meta in identity_map.items():
            if meta.get("name") != name:
                continue
            if (str(name), str(gid)) in dismissed_multicam:
                continue
            cameras_seen = []
            per_cam_frames = {}
            total = 0
            best_sim = 0.0
            crop_url = None
            for cam in CAMERAS:
                cam_id = cam["camera_id"]
                cam_frames = frames.get(cam_id) or {}
                seen = sum(
                    1 for people in cam_frames.values()
                    if any(str(p.get("global_id")) == str(gid) and p.get("name") == name for p in people)
                )
                if seen:
                    cameras_seen.append(cam_id)
                    per_cam_frames[cam_id] = seen
                    total += seen
                    if crop_url is None:
                        for fk, people in cam_frames.items():
                            for p in people:
                                if str(p.get("global_id")) != str(gid) or p.get("name") != name:
                                    continue
                                try:
                                    best_sim = max(best_sim, float(p.get("name_similarity") or 0.0))
                                except (TypeError, ValueError):
                                    pass
                                crop = crops_dir / cam_id / f"track{p['id']}_frame{fk}.jpg"
                                if crop.exists():
                                    crop_url = _data_url(crop)
                                    break
                            if crop_url:
                                break
            if cameras_seen:
                alerts.append({
                    "id": f"mc:{name}:{gid}",
                    "person": name,
                    "flag": info["flag"],
                    "details": info["details"],
                    "source": "multi-camera run",
                    "camera_id": cameras_seen[0],
                    "cameras": cameras_seen,
                    "cameras_frames": per_cam_frames,
                    "frames": total,
                    "similarity": round(best_sim, 4),
                    "gid": int(gid),
                    "crop_url": crop_url,
                    "time": run_time,
                })

    for ev in _read_alerts():
        ev = _normalize_alert_event(ev)
        alert_id = _alert_event_id(ev)
        if alert_id in dismissed_events:
            continue
        ev["id"] = alert_id
        alerts.append(ev)

    alerts.sort(key=lambda a: a.get("time") or "", reverse=True)
    return JSONResponse({"alerts": alerts})


@app.delete("/api/alerts/{alert_id}")
def api_delete_alert(alert_id: str):
    """Dismiss an alert after it has been handled.

    * Alert events (single-video / live-cam runs) are removed from
      outputs/alerts.json so they never come back.
    * Multi-camera run alerts (id "mc:<person>:<gid>") are recorded in
      outputs/dismissed_alerts.json and stay hidden for that person while their
      global identity is unchanged.
    """
    dismissed = _read_dismissed()
    removed = False

    if alert_id.startswith("mc:"):
        parts = alert_id.split(":", 2)
        if len(parts) == 3:
            key = [parts[1], parts[2]]
            multicam = dismissed.get("multicam", [])
            if key not in multicam:
                multicam.append(key)
                dismissed["multicam"] = multicam
            removed = True
    else:
        alerts = _read_alerts()
        kept = []
        for ev in alerts:
            if _alert_event_id(_normalize_alert_event(ev)) == alert_id:
                removed = True
                continue
            kept.append(ev)
        if len(kept) != len(alerts):
            ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ALERTS_PATH, "w", encoding="utf-8") as f:
                json.dump(kept[:200], f, indent=2, ensure_ascii=False)

    if removed:
        _write_dismissed(dismissed)
    return JSONResponse({"deleted": removed, "alert_id": alert_id})


# ── jobs ──────────────────────────────────────────────────────────────────────
@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(
            {
                "id": job["id"],
                "kind": job["kind"],
                "status": job["status"],
                "percent": job["percent"],
                "message": job["message"],
                "log": job["log"][-60:],
                "result": job["result"],
                "error": job["error"],
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
