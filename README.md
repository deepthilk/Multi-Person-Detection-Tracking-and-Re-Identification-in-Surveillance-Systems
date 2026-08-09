## Multi-Person Detection, Tracking and Re-Identification in Surveillance Systems

Detect, track and re-identify persons across video frames and cameras using
YOLOv8 + DeepSORT + a fine-tuned appearance Re-ID model fused with face cues.
A web dashboard drives the whole flow: register known people, process recorded
or live webcam footage, run the integrated multi-camera pipeline, search videos
by face, and get alerts for flagged persons.

## Current Project Flow

```
Registration              Processing                    Outputs
──────────────────        ─────────────────────────     ──────────────────────
person photos   ──►       ─► [1] Detection (YOLOv8)     rendered annotated video
                           ► [2] Tracking (DeepSORT)    identity matches
identity_db.json      ─►   ► [3] Re-ID (appearance      alerts for flagged
face_db.json               │      + face cues)          persons
                           ► [4] Identify vs.           face-search index +
                              registered persons        highlight clips
                           ► [5] Render                 contact sheets
```

The web dashboard orchestrates everything. Processing runs in a background
worker so the dashboard stays responsive.

## Required Assets

The repository intentionally excludes generated/runtime files. To run it you need:

- `models/yolov8s.pt` — YOLOv8 detection weights
- `reidentification/weights/best_model_cloth.pth` — Re-ID model weights (loaded by `reid_main.py`)
- `input/video1.mp4` … `input/video4.mp4` — default sample videos
- `venv/` — the Python virtual environment (dependencies from `requirements.txt`)

Your registered-person data lives under `outputs/registration/`
(`identity_db.json`, `face_db.json`, `images/<person>/`) and must not be deleted.

## Setup

```bash
# 1. Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the dashboard
python web/server.py
```

Open http://localhost:8000 (server runs on port 8000).

## Using the Web Dashboard

1. **Register people** — add a person's name and a few clear, face-visible
   photos. Photos are embedded into the identity DB (body) and face DB (ArcFace)
   under `outputs/registration/`.
2. **Process a video** — upload recorded `.mp4/.avi/.mov/.mkv/.webm` clips and
   run detect → track → re-ID → identify → render. Identical videos are served
   from cache. The rendered clip shows green boxes + names and registered
   persons are matched.
3. **Live cam** — record a short clip (3–30 s) from the webcam, then run the
   same pipeline on it. Recorded clips are saved to `web/uploads/`.
4. **Integrated pipeline** — run the full multi-camera pipeline on the input
   videos from the UI (same as `run_integrated_pipeline.py` below), producing
   cross-camera global identities.
5. **Face search** — pick a registered person and search the videos by face
   encoding. Each match produces a highlight clip and a verification contact
   sheet (`outputs/face_search/`).
6. **Alerts** — flagged (missing) persons raise alerts on the dashboard. Mark
   an alert "Handled" to dismiss it permanently.

## Command-Line Usage

```bash
# Run the full integrated pipeline on input/video1..4.mp4
python run_integrated_pipeline.py --device cpu

# Restrict to the first N frames per video (quick test)
python run_integrated_pipeline.py --device cpu --max-frames 60

# Run the multi-camera pipeline
python run_multicam.py
```

## Directory Structure

```
├── detection/            YOLOv8 person detection (detect_module.py)
├── tracking/             DeepSORT tracking (track_module.py)
├── reidentification/     Re-ID pipeline, face cues, cross-camera matching,
│                         face search (reid_main.py, face_search.py, …)
├── registration/         Identity + face databases (identity_db.py, face_db.py)
├── multicamera/          Multi-camera config and pipeline (camera_config.py,
│                         multi_cam_pipeline.py, stream_manager.py)
├── web/                  FastAPI dashboard (server.py), video worker
│                         (video_worker.py), static UI (index.html, app.js)
├── models/               yolov8s.pt
├── input/                sample videos
├── outputs/              generated results + registration data
├── run_integrated_pipeline.py   full pipeline CLI
├── run_multicam.py              multi-camera CLI
├── utils.py                     shared helpers + video renderer
└── requirements.txt
```

## Where Things Are Stored

| Item                              | Location                                    |
|-----------------------------------|---------------------------------------------|
| Uploaded / recorded clips         | `web/uploads/`                              |
| Per-job results (detections, tracking, re-ID, result JSON, rendered video) | `web/outputs/` |
| Rendered videos + camera sources  | `outputs/rendered/`                         |
| Registration (identity + face DB, person photos) | `outputs/registration/` |
| Face-search indexes + clips + sheets | `outputs/face_search/`                   |
| Alerts                            | `outputs/alerts.json`                       |

All of these are regenerable (except `outputs/registration/`, which holds your
registered people).

## Troubleshooting

- **"No webcam available"** — the live-cam flow needs an attached camera; use
  the recorded-video upload instead.
- **Slow on CPU** — pass `--device cpu` explicitly and reduce `--max-frames`
  for quick runs.
- **Missing model weights** — ensure `models/yolov8s.pt` and
  `reidentification/weights/best_model_cloth.pth` are present.
