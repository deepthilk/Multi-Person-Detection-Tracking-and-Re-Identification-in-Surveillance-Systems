"""
Person Re-Identification module.

Contains the production Re-ID pipeline (per-camera identity assignment with
re-appearance handling), cross-camera identity matching, track-level
clustering, and the face-cue helpers that fuse face embeddings with body
descriptors for robust naming across cameras.

Public API:
    from reidentification.reid_main import run_reid_pipeline
    from reidentification.cross_camera_match import run_cross_camera_matching
    from reidentification.track_cluster import run_track_level_pipeline
"""
