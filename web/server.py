from pathlib import Path
import shutil
import sys
import uuid
from typing import Dict

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from detection.detect_module import run_detection
from tracking.track_module import run_tracking
from reidentification.reid_main import run_reid_pipeline
from utils import render_reid_video


WEB_DIR = ROOT_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
UPLOAD_DIR = WEB_DIR / "uploads"
OUTPUT_DIR = WEB_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Re-ID Cinematic Web")
JOBS: Dict[str, dict] = {}


def _ensure_artifact(path: Path, label: str):
    if not path.exists():
        raise RuntimeError(f"{label} not created")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{label} is empty")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


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
            conf_threshold=0.6,  # Higher = faster but might miss people
            min_height=50,
            min_area_ratio=0.001,
            device=device,
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
