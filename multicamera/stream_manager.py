"""
Multi-Camera Stream Manager
============================

Responsible for:
  - Opening multiple camera/video sources (simulated CCTV cameras).
  - Reading frames from each source in lock-step ("synchronized" ticks).
  - Attaching a frame number + timestamp to each tick.
  - Handling sources of different length / fps gracefully.

This module has NO dependency on detection, tracking or re-id code, so it
can be developed, tested and merged independently of the rest of the team.
"""

import time
import logging
from datetime import datetime, timedelta

import cv2

logger = logging.getLogger(__name__)


class CameraSource:
    """Wraps a single cv2.VideoCapture (file, webcam index, or RTSP URL)."""

    def __init__(self, camera_id: str, source):
        self.camera_id = camera_id
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise IOError(f"Camera '{camera_id}': failed to open source '{source}'")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.fps <= 0:
            self.fps = 25.0

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        self.exhausted = False

        logger.info(
            f"✅ Camera '{camera_id}' opened  source={source}  fps={self.fps:.2f}  "
            f"frames={self.frame_count if self.frame_count else 'unknown'}"
        )

    def read(self):
        """Read the next frame. Returns None once the source is exhausted."""
        if self.exhausted:
            return None

        ret, frame = self.cap.read()
        if not ret:
            self.exhausted = True
            logger.info(f"Camera '{self.camera_id}' stream ended")
            return None

        return frame

    def release(self):
        self.cap.release()


class MultiCameraStreamManager:
    """
    Opens several CameraSource objects and steps through them together,
    producing one synchronized "tick" per call: a frame number plus a
    dict of {camera_id: frame_or_None}.

    Synchronization strategy:
      Each camera is advanced by exactly one frame per tick ("lock-step").
      This is the right model for simulating CCTV cameras from video files
      recorded independently. For live cameras with different frame rates,
      the timestamp attached to each frame (derived from that camera's own
      fps) is what downstream code should use for true time alignment.
    """

    def __init__(self, camera_configs, start_time: datetime = None):
        """
        Args:
            camera_configs: list of dicts like
                [{"camera_id": "cam1", "source": "input/video1.mp4"}, ...]
            start_time: wall-clock time to treat as frame_number=0 for every
                camera (defaults to "now"). Used only to generate realistic
                timestamps in the output records.
        """
        if not camera_configs:
            raise ValueError("camera_configs must contain at least one camera")

        self.start_time = start_time or datetime.now()
        self.cameras = {}
        for cfg in camera_configs:
            cam_id = cfg["camera_id"]
            if cam_id in self.cameras:
                raise ValueError(f"Duplicate camera_id in config: '{cam_id}'")
            self.cameras[cam_id] = CameraSource(cam_id, cfg["source"])

        self.frame_number = 0

    def all_exhausted(self) -> bool:
        return all(cam.exhausted for cam in self.cameras.values())

    def frame_timestamp(self, camera_id: str, frame_number: int) -> str:
        """ISO-8601 timestamp for a given camera + frame number, derived
        from that camera's own fps so cameras with different frame rates
        still get meaningful, comparable wall-clock timestamps."""
        cam = self.cameras[camera_id]
        seconds_offset = frame_number / cam.fps
        return (self.start_time + timedelta(seconds=seconds_offset)).isoformat()

    def synchronized_frames(self):
        """
        Generator. Yields (frame_number, {camera_id: frame_or_None}) tuples
        until every camera is exhausted. A camera that has already ended
        yields None for every subsequent tick so the others keep going.
        """
        while not self.all_exhausted():
            self.frame_number += 1
            frames = {}
            for cam_id, cam in self.cameras.items():
                frames[cam_id] = cam.read()

            # If this tick produced no frames at all (every camera just
            # became exhausted on this exact read), stop instead of
            # yielding a useless all-None tick.
            if all(f is None for f in frames.values()):
                break

            yield self.frame_number, frames

    def release(self):
        for cam in self.cameras.values():
            cam.release()
        logger.info("All camera sources released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
