# Multi-Camera Processing and Video Stream Management

**Status:** Phase 2 module — independent, no changes required in
`detection/`, `tracking/`, `reidentification/`, or `web/`.

## What this module does

Reads multiple camera/video streams at once (simulating multiple CCTV
cameras with separate video files), runs the existing Detection +
Tracking pipeline **independently per camera**, and produces standardized,
timestamped person records ready for the Re-ID module.

```
input/video1.mp4 ─┐
input/video2.mp4 ─┼─> MultiCameraStreamManager ─> per-camera Detector+Tracker ─> standardized records ─> Re-ID
input/video3.mp4 ─┘
```

## Files

| File | Purpose |
|---|---|
| `camera_config.py` | List of cameras + output paths. Edit this to add/remove cameras. Owned by this module — no one else needs to touch it. |
| `stream_manager.py` | Opens N video sources, steps them in lock-step, produces `(frame_number, {camera_id: frame})` ticks, per-camera timestamps, handles streams of different length/fps. |
| `multi_cam_pipeline.py` | Runs a separate `PersonDetector` + `PersonTracker` per camera, extracts person crops, builds the standardized output. |
| `run_multicam.py` (repo root) | CLI entry point. Separate from `main.py` on purpose — zero merge-conflict risk. |

## Output contract (for the Re-ID teammate)

1. **`outputs/multicam/person_records.json`** — flat list of every detected
   person across all cameras:
   ```json
   {
     "camera_id": "cam1",
     "track_id": 7,
     "frame_number": 145,
     "timestamp": "2026-07-18T10:12:33.500000",
     "bbox": [x1, y1, x2, y2],
     "crop_path": "outputs/multicam/crops/cam1/track7_frame145.jpg"
   }
   ```
   `(camera_id, track_id)` together are the unique key — `track_id` alone
   is only unique **within** one camera.

2. **`outputs/multicam/crops/<camera_id>/track<id>_frame<n>.jpg`** — the
   actual cropped person image, already saved to disk.

3. **`outputs/multicam/tracking/<camera_id>_tracking.json`** — same schema
   as the existing single-camera `outputs/tracking.json`
   (`{frame_number: [{"id":.., "bbox":[..]}]}`). This means the existing
   `reidentification.reid_main.run_reid_pipeline(video_path, tracking_json_path, output_json_path)`
   can be called **unmodified**, once per camera, e.g.:
   ```python
   run_reid_pipeline("input/video1.mp4",
                      "outputs/multicam/tracking/cam1_tracking.json",
                      "outputs/multicam/reid/cam1_reid.json")
   ```

4. **Live callback** — for real-time integration instead of waiting on
   files, pass `on_person_detected=your_function` to
   `run_multicamera_pipeline()`; it's called with each record the instant
   it's produced.

## Running it

```bash
# Edit multicamera/camera_config.py first so CAMERAS points at real videos
python run_multicam.py --device cpu
python run_multicam.py --device cuda --conf-threshold 0.4
python run_multicam.py --max-frames 200   # quick smoke test
```

## Why this design avoids merge conflicts

- New folder (`multicamera/`) + new root script (`run_multicam.py`) — no
  existing file is modified.
- Own `camera_config.py` instead of touching the shared `config.py`.
- Imports `PersonDetector` / `PersonTracker` as-is; does not modify
  `detection/` or `tracking/`.
- Does not import anything from `reidentification/` or `web/`.
