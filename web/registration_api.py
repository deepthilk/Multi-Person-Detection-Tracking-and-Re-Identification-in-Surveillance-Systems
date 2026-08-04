"""
Registration & Auto-fix REST API — the backend contract for Pranjali's UI.

Exposes the registration module (add/list/delete/verify) plus the auto-fix
candidate workflow (scan videos -> show crops -> pick one -> re-register).

Slow operations (auto-fix video scanning, verification) run as background
jobs: the frontend calls the POST endpoint, gets a `job_id`, and polls
`GET /api/autofix/{job_id}` until status == "completed".

See API_CONTRACT.md for the full request/response reference.
"""

import shutil
import sys
import uuid
from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from registration.autofix import get_autofix_dir, get_detector, scan_video_for_candidates
from registration.db_config import SEARCH_SETTINGS
from registration.embedder import embed_image
from registration.identity_db import IdentityDatabase
from registration.register_person import register_person

router = APIRouter(prefix="/api")

UPLOAD_DIR = ROOT_DIR / "web" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB — an image bigger than this is an error

# name -> job dict (mirrors the pattern already used in server.py)
AUTOFIX_JOBS: Dict[str, dict] = {}

# Shared detector is built lazily and cached — loading YOLO on every request
# is slow.
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = get_detector()
    return _detector


def _is_readable_image(path: Path) -> bool:
    """Reject corrupt / non-image uploads BEFORE they reach the embedder."""
    try:
        import cv2
        img = cv2.imread(str(path))
        return img is not None and img.size > 0
    except Exception:
        return False


def _clamp(value: int, lo: int, hi: int, name: str) -> int:
    """Bound an integer form field so a client cannot start pathological jobs
    (e.g. 1e9 augmentation passes or samples)."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"{name} must be an integer")
    if not lo <= value <= hi:
        raise HTTPException(status_code=400, detail=f"{name} must be between {lo} and {hi}")
    return value


def _require_person(db, name) -> dict:
    record = db.get_person(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Person '{name}' not found")
    return record


def _save_upload(file: UploadFile, prefix: str) -> Path:
    """Save an uploaded file after validating type, size, and readability.
    Invalid uploads are rejected with a clear 4xx and never reach the
    embedder / database."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}' — use jpg, jpeg, png, or bmp",
        )

    safe_name = Path(file.filename).name
    dest = UPLOAD_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}_{safe_name}"
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    size = dest.stat().st_size
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes)")
    if size > MAX_UPLOAD_BYTES:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file exceeds the 20 MB limit")
    if not _is_readable_image(dest):
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"'{file.filename}' is not a readable image (corrupt or wrong format)",
        )
    return dest


def _person_summary(db, name):
    rec = db.get_person(name)
    return {
        "name": name,
        "num_images": rec["metadata"]["num_images"],
        "registered_at": rec["metadata"]["registered_at"],
        "last_updated": rec["metadata"].get("last_updated"),
    }


# ── videos ────────────────────────────────────────────────────────────────────

@router.get("/videos")
def list_videos():
    """List the configured CCTV cameras so the UI can let the user choose
    which video(s) to scan during auto-fix."""
    from multicamera.camera_config import CAMERAS
    out = []
    for cfg in CAMERAS:
        src = cfg["source"]
        out.append({
            "camera_id": cfg["camera_id"],
            "source": src,
            "exists": Path(src).exists(),
        })
    return {"videos": out}


# ── registration CRUD ─────────────────────────────────────────────────────────

@router.get("/registration")
def list_persons():
    db = IdentityDatabase()
    return {"persons": [_person_summary(db, n) for n in db.list_persons()]}


@router.get("/registration/{name}")
def get_person(name: str):
    db = IdentityDatabase()
    rec = _require_person(db, name)
    return {
        **_person_summary(db, name),
        "image_paths": rec["metadata"].get("image_paths", []),
        "match_threshold": SEARCH_SETTINGS["match_threshold"],
    }


@router.post("/registration")
def add_person(name: str = Form(...),
               files: list[UploadFile] = File(...),
               overwrite: bool = Form(False),
               augmentations: int = Form(5)):
    """Register a person from one or more uploaded photos.

    Uses the domain-robust embedder (video simulation + augmentation
    averaging). If `overwrite` is true, replaces the person's existing
    photos instead of appending.
    """
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    if not files:
        raise HTTPException(status_code=400, detail="At least one image required")
    augmentations = _clamp(augmentations, 0, 30, "augmentations")

    paths = [_save_upload(f, "reg") for f in files]
    try:
        record = register_person(name.strip(), [str(p) for p in paths],
                                 overwrite=overwrite,
                                 num_augmentations=augmentations)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"registered": True, "person": _person_summary(IdentityDatabase(), name)}


@router.delete("/registration/{name}")
def delete_person(name: str):
    db = IdentityDatabase()
    if not db.delete_person(name):
        raise HTTPException(status_code=404, detail=f"Person '{name}' not found")
    return {"deleted": True, "name": name}


# ── verify (low-confidence check) ─────────────────────────────────────────────

@router.post("/registration/{name}/verify")
async def verify_person(name: str, test_image: UploadFile = File(...),
                        augmentations: int = Form(5)):
    """Embed an uploaded test image (typically a frame/crop from the video)
    and compare it against the registered person's embedding.

    Returns similarity, the match threshold, and whether the registration is
    usable. Below-threshold scores are what trigger the auto-fix flow.
    """
    db = IdentityDatabase()
    rec = _require_person(db, name)
    ref_avg = np.array(rec["average_embedding"], dtype=np.float32)
    augmentations = _clamp(augmentations, 0, 30, "augmentations")

    path = _save_upload(test_image, "verify")
    feat = embed_image(str(path), num_augmentations=augmentations)
    if feat is None:
        raise HTTPException(status_code=422, detail="Could not embed the uploaded test image")

    from registration.autofix import cosine_sim
    sim = cosine_sim(feat, ref_avg)
    threshold = SEARCH_SETTINGS["match_threshold"]

    per_photo = []
    for i, emb in enumerate(rec["embeddings"]):
        per_photo.append({"photo": i + 1, "similarity": round(cosine_sim(feat, emb), 4)})

    return {
        "name": name,
        "similarity": round(sim, 4),
        "threshold": threshold,
        "passed": sim >= threshold,
        "gap": round(max(0.0, threshold - sim), 4),
        "per_photo": per_photo,
        "suggest_autofix": sim < threshold,
    }


# ── auto-fix candidate scan (background job) ──────────────────────────────────


class AutofixRequest(BaseModel):
    videos: list[str]
    samples: int = 20
    top: int = 5
    augmentations: int = 5


class AutofixConfirmRequest(BaseModel):
    job_id: str
    index: int
    augmentations: int = 5


@router.post("/registration/{name}/autofix")
async def start_autofix(name: str, body: AutofixRequest):
    """Start scanning the given videos for candidate crops of `name`.

    Runs as a background job because detection + embedding is slow. Returns
    a `job_id`; poll `GET /api/autofix/{job_id}` for results.

    Body (JSON):
        {"videos": ["input/video1.mp4", "input/video2.mp4"],
         "samples": 20, "top": 5, "augmentations": 5}
    """
    db = IdentityDatabase()
    rec = _require_person(db, name)

    if not body.videos:
        raise HTTPException(status_code=400, detail="At least one video required")

    samples = _clamp(body.samples, 1, 200, "samples")
    top = _clamp(body.top, 1, 50, "top")
    augmentations = _clamp(body.augmentations, 0, 30, "augmentations")

    for v in body.videos:
        if not Path(v).exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {v}")

    job_id = uuid.uuid4().hex[:10]
    AUTOFIX_JOBS[job_id] = {
        "job_id": job_id,
        "name": name,
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "candidates": [],
    }

    import asyncio
    asyncio.create_task(_run_autofix_job(
        job_id, name, body.videos, samples, top, augmentations))
    return {"job_id": job_id, "status": "queued"}


async def _run_autofix_job(job_id, name, videos, samples, top, augmentations):
    try:
        job = AUTOFIX_JOBS[job_id]
        job.update({"status": "running", "progress": 5, "message": "Loading database"})

        db = IdentityDatabase()
        rec = db.get_person(name)
        ref_avg = np.array(rec["average_embedding"], dtype=np.float32)
        detector = _get_detector()
        out_dir = get_autofix_dir()

        all_candidates = []
        for i, vid in enumerate(videos):
            job.update({
                "progress": 5 + int(80 * i / len(videos)),
                "message": f"Scanning {Path(vid).name}",
            })
            all_candidates.extend(scan_video_for_candidates(
                vid, detector, ref_avg, out_dir, samples, augmentations))

        all_candidates.sort(key=lambda x: x[0], reverse=True)
        top_cands = all_candidates[:top]

        threshold = SEARCH_SETTINGS["match_threshold"]
        candidates = []
        for idx, (sim, vid_name, fid, x1, y1, w, h, crop_path) in enumerate(top_cands):
            candidates.append({
                "index": idx,
                "video": vid_name,
                "frame": fid,
                "bbox": [x1, y1, x1 + w, y1 + h],
                "similarity": round(sim, 4),
                "above_threshold": sim >= threshold,
                "crop_url": f"/api/autofix/{job_id}/crop/{idx}",
                "crop_path": crop_path,
            })

        job.update({
            "status": "completed",
            "progress": 100,
            "message": f"Found {len(candidates)} candidate(s)",
            "candidates": candidates,
            "threshold": threshold,
        })
    except Exception as exc:
        AUTOFIX_JOBS[job_id].update({
            "status": "error",
            "progress": 100,
            "message": str(exc),
        })


@router.get("/autofix/{job_id}")
def get_autofix(job_id: str):
    if job_id not in AUTOFIX_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(AUTOFIX_JOBS[job_id])


@router.get("/autofix/{job_id}/crop/{index}")
def get_autofix_crop(job_id: str, index: int):
    job = AUTOFIX_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    cands = job.get("candidates", [])
    pool = job.get("_pool", [])
    if cands and 0 <= index < len(cands):
        path = Path(cands[index]["crop_path"])
    elif pool and 0 <= index < len(pool):
        path = Path(pool[index])
    else:
        raise HTTPException(status_code=404, detail="Candidate index out of range")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Crop file missing")
    return FileResponse(str(path), media_type="image/jpeg")


@router.post("/registration/{name}/autofix/confirm")
def confirm_autofix(name: str, body: AutofixConfirmRequest):
    """Register `name` from the crop the user selected in the UI gallery.

    The crop is embedded and stored as the person's (only) registration
    image — replacing the photo that failed the low-confidence check.

    Body (JSON):
        {"job_id": "abc123", "index": 2}
    """
    job = AUTOFIX_JOBS.get(body.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    cands = job.get("candidates", [])
    if body.index < 0 or body.index >= len(cands):
        raise HTTPException(status_code=404, detail="Candidate index out of range")

    cand = cands[body.index]
    crop_path = cand["crop_path"]
    if not Path(crop_path).exists():
        raise HTTPException(status_code=404, detail="Crop file missing")

    try:
        register_person(name, [crop_path], overwrite=True,
                        num_augmentations=_clamp(body.augmentations, 0, 30, "augmentations"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "registered": True,
        "name": name,
        "source_video": cand["video"],
        "frame": cand["frame"],
        "similarity": cand["similarity"],
        "crop_url": cand["crop_url"],
    }


# ── multi-person auto-fix (all registered persons, one scan) ─────────────────


class AutofixAllRequest(BaseModel):
    videos: list[str]
    samples: int = 20
    top: int = 5
    augmentations: int = 5


class AutofixAllConfirmRequest(BaseModel):
    job_id: str
    name: str
    index: int
    augmentations: int = 5


@router.post("/autofix/all")
async def start_autofix_all(body: AutofixAllRequest):
    """Scan the given videos once and produce top-`top` candidate crops for
    EVERY registered person. Use this when a session returned poor matches —
    one pass over the videos, re-register each person from a picked crop.

    Body (JSON):
        {"videos": ["input/video1.mp4"], "samples": 20, "top": 5, "augmentations": 5}

    Response: {"job_id": "...", "status": "queued"} — poll
    GET /api/autofix/{job_id}; the completed job carries a `persons` array,
    one entry per registered person with `candidates` (top 5, crop_urls).
    """
    if not body.videos:
        raise HTTPException(status_code=400, detail="At least one video required")
    samples = _clamp(body.samples, 1, 200, "samples")
    top = _clamp(body.top, 1, 50, "top")
    augmentations = _clamp(body.augmentations, 0, 30, "augmentations")
    for v in body.videos:
        if not Path(v).exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {v}")

    db = IdentityDatabase()
    names = db.list_persons()
    if not names:
        raise HTTPException(status_code=409, detail="No registered persons to auto-fix")

    ref_embeddings = {
        name: np.array(db.get_person(name)["average_embedding"], dtype=np.float32)
        for name in names
    }

    job_id = uuid.uuid4().hex[:10]
    AUTOFIX_JOBS[job_id] = {
        "job_id": job_id,
        "mode": "all",
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "persons": [],
    }

    import asyncio
    asyncio.create_task(_run_autofix_all_job(
        job_id, body.videos, ref_embeddings, samples, top, augmentations))
    return {"job_id": job_id, "status": "queued"}


async def _run_autofix_all_job(job_id, videos, ref_embeddings, samples, top, augmentations):
    try:
        job = AUTOFIX_JOBS[job_id]
        job.update({"status": "running", "progress": 5, "message": "Loading detector"})

        detector = _get_detector()
        out_dir = get_autofix_dir()
        threshold = SEARCH_SETTINGS["match_threshold"]

        def on_video(done, total):
            job.update({
                "progress": 5 + int(85 * done / max(1, total)),
                "message": f"Scanned {done}/{total} video(s)",
            })

        from registration.autofix import scan_videos_for_all_persons
        best = scan_videos_for_all_persons(
            videos, detector, ref_embeddings, out_dir,
            samples=samples, augmentations=augmentations, top=top,
            on_video=on_video)

        # One crop pool for the whole job so the same crop shared by several
        # persons is stored once and served through /crop/{pool_index}.
        pool = []
        pool_index = {}
        for lst in best.values():
            for cand in lst:
                p = cand[-1]
                if p not in pool_index:
                    pool_index[p] = len(pool)
                    pool.append(p)

        persons = []
        for name in ref_embeddings:
            candidates = []
            for idx, (sim, vid_name, fid, x1, y1, w, h, crop_path) in enumerate(best[name]):
                pi = pool_index[crop_path]
                candidates.append({
                    "index": idx,
                    "pool_index": pi,
                    "video": vid_name,
                    "frame": fid,
                    "bbox": [x1, y1, x1 + w, y1 + h],
                    "similarity": round(sim, 4),
                    "above_threshold": sim >= threshold,
                    "crop_url": f"/api/autofix/{job_id}/crop/{pi}",
                    "crop_path": crop_path,
                })
            persons.append({"name": name, "candidates": candidates})

        job.update({
            "status": "completed",
            "progress": 100,
            "message": f"Top {top} candidate(s) for {len(persons)} person(s)",
            "persons": persons,
            "threshold": threshold,
            "_pool": pool,
        })
    except Exception as exc:
        AUTOFIX_JOBS[job_id].update({
            "status": "error",
            "progress": 100,
            "message": str(exc),
        })


@router.post("/autofix/all/confirm")
def confirm_autofix_all(body: AutofixAllConfirmRequest):
    """Re-register one person from a crop picked in the batch gallery.

    Body (JSON):
        {"job_id": "abc123", "name": "Alice", "index": 2}
    """
    job = AUTOFIX_JOBS.get(body.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    persons = job.get("persons", [])
    person = next((p for p in persons if p["name"] == body.name), None)
    if person is None:
        raise HTTPException(status_code=404, detail=f"'{body.name}' not in this job")

    cands = person.get("candidates", [])
    if body.index < 0 or body.index >= len(cands):
        raise HTTPException(status_code=404, detail="Candidate index out of range")

    cand = cands[body.index]
    crop_path = cand["crop_path"]
    if not Path(crop_path).exists():
        raise HTTPException(status_code=404, detail="Crop file missing")

    try:
        register_person(body.name, [crop_path], overwrite=True,
                        num_augmentations=_clamp(body.augmentations, 0, 30, "augmentations"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "registered": True,
        "name": body.name,
        "source_video": cand["video"],
        "frame": cand["frame"],
        "similarity": cand["similarity"],
        "crop_url": cand["crop_url"],
    }
