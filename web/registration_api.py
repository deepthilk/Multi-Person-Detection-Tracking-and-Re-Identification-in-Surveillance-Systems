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


def _require_person(db, name) -> dict:
    record = db.get_person(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Person '{name}' not found")
    return record


def _save_upload(file: UploadFile, prefix: str) -> Path:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")
    safe_name = Path(file.filename).name
    dest = UPLOAD_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}_{safe_name}"
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
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
        job_id, name, body.videos, body.samples, body.top, body.augmentations))
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
    if index < 0 or index >= len(cands):
        raise HTTPException(status_code=404, detail="Candidate index out of range")
    path = Path(cands[index]["crop_path"])
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

    register_person(name, [crop_path], overwrite=True,
                    num_augmentations=body.augmentations)

    return {
        "registered": True,
        "name": name,
        "source_video": cand["video"],
        "frame": cand["frame"],
        "similarity": cand["similarity"],
        "crop_url": cand["crop_url"],
    }
