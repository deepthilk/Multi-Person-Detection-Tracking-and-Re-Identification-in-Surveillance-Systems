"""
Camera configuration for the Multi-Camera Processing Module.

This file is OWNED by the multi-camera module and is intentionally
kept separate from the shared config.py to avoid merge conflicts with
teammates who also edit config.py (detection / tracking / re-id settings).

Each entry simulates one CCTV camera using a video file. Replace the
'source' paths with real video files (or an RTSP URL / webcam index for a
live camera) before running the pipeline.
"""

# ==============================================================================
# CAMERA DEFINITIONS
# ==============================================================================
# 'source' can be:
#   - a path to a video file (used to simulate a CCTV camera)
#   - an integer (webcam index, e.g. 0)
#   - an RTSP/HTTP stream URL (for real IP cameras)
CAMERAS = [
    {"camera_id": "cam1", "source": "input/video1.mp4"},
    {"camera_id": "cam2", "source": "input/video2.mp4"},
    {"camera_id": "cam3", "source": "input/video3.mp4"},
]

# ==============================================================================
# MULTI-CAMERA PIPELINE SETTINGS
# ==============================================================================
MULTICAM_SETTINGS = {
    # Where standardized outputs are written
    "output_dir": "outputs/multicam",
    "crops_dir": "outputs/multicam/crops",
    "tracking_dir": "outputs/multicam/tracking",
    "records_json": "outputs/multicam/person_records.json",

    # If a camera stream runs out of frames before the others, keep the
    # remaining streams going instead of stopping the whole pipeline.
    "stop_when_all_exhausted": True,

    # Minimum crop size (pixels) to bother saving/forwarding to Re-ID.
    "min_crop_width": 20,
    "min_crop_height": 40,

    # JPEG quality for saved person crops.
    "crop_jpeg_quality": 90,
}
