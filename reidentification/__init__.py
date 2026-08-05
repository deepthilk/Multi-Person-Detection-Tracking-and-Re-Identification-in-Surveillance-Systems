"""
Person Re-Identification (Re-ID) Module
========================================

Core components:
  - reid_main.py     : Frame-by-frame Re-ID engine (ReIDEngine, MultiCueExtractor)
  - face_cue.py      : Face-based Re-ID cue (dlib/face_recognition, optional)
  - insight_face.py  : InsightFace (ArcFace) face extractor for registration/search
  - cross_camera_match.py : Cross-camera identity matching (Hungarian assignment)
  - track_cluster.py : Offline track-level clustering (replaces per-frame decisions)

Sub-modules:
  - model/           : Re-ID model architectures (ResNetReIDBackbone, OSNet loader)
  - training/        : Fine-tuning and evaluation on Market-1501
  - weights/         : Model weight files (best_model.pth, last_checkpoint.pth)
"""

from reidentification.reid_main import ReIDEngine, run_reid_pipeline, diagnose
from reidentification.face_cue import FaceCueExtractor
from reidentification.insight_face import InsightFaceExtractor
from reidentification.cross_camera_match import CrossCameraMatcher, run_cross_camera_matching
from reidentification.track_cluster import run_track_level_pipeline

__all__ = [
    "ReIDEngine",
    "run_reid_pipeline",
    "diagnose",
    "FaceCueExtractor",
    "InsightFaceExtractor",
    "CrossCameraMatcher",
    "run_cross_camera_matching",
    "run_track_level_pipeline",
]
