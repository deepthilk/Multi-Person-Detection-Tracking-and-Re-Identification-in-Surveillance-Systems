"""
Multi-Camera Processing Pipeline
==================================

Ties together:
  - multicamera.stream_manager  (reads & synchronizes camera streams)
  - detection.detect_module     (existing YOLOv8 PersonDetector — untouched)
  - tracking.track_module       (existing DeepSORT PersonTracker — untouched)

...and produces STANDARDIZED per-person records ready to hand off to the
Re-ID module:

    {
        "camera_id":    "cam1",
        "track_id":     3,
        "frame_number": 145,
        "timestamp":    "2026-07-18T10:12:33.500000",
        "bbox":         [x1, y1, x2, y2],
        "crop_path":    "outputs/multicam/crops/cam1/track3_frame145.jpg"
    }

Design notes for integration:
  - A SEPARATE detector + tracker instance is created per camera. Track IDs
    from DeepSORT are only unique *within* a camera, so a (camera_id,
    track_id) pair is the unique key — never track_id alone. This is the
    contract the Re-ID module should rely on.
  - Per-camera outputs are ALSO written in the exact schema that
    reidentification/reid_main.py already expects (frame_id -> [{"id","bbox"}]),
    saved to outputs/multicam/tracking/<camera_id>_tracking.json. That means
    the existing `run_reid_pipeline(video_path, tracking_json_path, ...)`
    can be called unmodified, once per camera, with no changes required to
    the Re-ID teammate's code.
  - This module does NOT import anything from reidentification/, so it has
    zero risk of merge conflicts with that teammate's branch.
"""

import json
import logging
from pathlib import Path

import cv2

from detection.detect_module import PersonDetector
from tracking.track_module import PersonTracker
from multicamera.stream_manager import MultiCameraStreamManager
from multicamera.camera_config import CAMERAS, MULTICAM_SETTINGS

logger = logging.getLogger(__name__)


def _safe_crop(frame, bbox, min_w=20, min_h=40):
    """Crop bbox out of frame, clamped to frame bounds. Returns None if the
    resulting crop is too small to be useful for Re-ID."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 - x1 < min_w or y2 - y1 < min_h:
        return None

    return frame[y1:y2, x1:x2]


class MultiCameraPipeline:
    """
    Runs detection + tracking independently per camera, frame-synchronized
    across cameras, and emits standardized person records.
    """

    def __init__(self, camera_configs=None, settings=None, device="cpu",
                 conf_threshold=0.35):
        self.camera_configs = camera_configs or CAMERAS
        self.settings = settings or MULTICAM_SETTINGS
        self.device = device
        self.conf_threshold = conf_threshold

        # One detector + tracker PER CAMERA — keeps track IDs, Kalman state
        # and appearance galleries fully independent between cameras.
        self.detectors = {}
        self.trackers = {}
        for cfg in self.camera_configs:
            cam_id = cfg["camera_id"]
            self.detectors[cam_id] = PersonDetector(
                conf_threshold=self.conf_threshold, device=self.device
            )
            self.trackers[cam_id] = PersonTracker()

        self.records = []                      # flat standardized output
        self.per_camera_tracking = {           # reid_main.py-compatible
            cfg["camera_id"]: {} for cfg in self.camera_configs
        }

        Path(self.settings["crops_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.settings["tracking_dir"]).mkdir(parents=True, exist_ok=True)

    def run(self, on_person_detected=None, max_frames=None):
        """
        Process all camera streams to completion.

        Args:
            on_person_detected: optional callback(record: dict) invoked the
                instant a standardized person record is produced — lets the
                Re-ID module (or a live dashboard) consume results in real
                time instead of waiting for the JSON file at the end.
            max_frames: optional cap on number of synchronized ticks, useful
                for quick local testing without processing full videos.

        Returns:
            (records, per_camera_tracking) — also written to disk.
        """
        with MultiCameraStreamManager(self.camera_configs) as stream_mgr:
            for frame_number, frames in stream_mgr.synchronized_frames():
                if max_frames and frame_number > max_frames:
                    break

                for cam_id, frame in frames.items():
                    if frame is None:
                        continue

                    self._process_camera_frame(
                        cam_id, frame, frame_number,
                        stream_mgr.frame_timestamp(cam_id, frame_number),
                        on_person_detected,
                    )

                if frame_number % 50 == 0:
                    logger.info(f"Tick {frame_number}: "
                                f"{len(self.records)} person records so far")

        self._save_outputs()
        return self.records, self.per_camera_tracking

    def _process_camera_frame(self, cam_id, frame, frame_number, timestamp,
                               on_person_detected):
        detector = self.detectors[cam_id]
        tracker = self.trackers[cam_id]

        detections = detector.detect(frame)
        tracks = tracker.update(frame, detections)

        # Keep per-camera tracking.json compatible with reid_main.py schema
        self.per_camera_tracking[cam_id][frame_number] = [
            {"id": t["id"], "bbox": t["bbox"]} for t in tracks
        ]

        for track in tracks:
            crop = _safe_crop(
                frame, track["bbox"],
                min_w=self.settings["min_crop_width"],
                min_h=self.settings["min_crop_height"],
            )
            if crop is None:
                continue

            crop_path = (
                f"{self.settings['crops_dir']}/{cam_id}/"
                f"track{track['id']}_frame{frame_number}.jpg"
            )
            Path(crop_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                crop_path, crop,
                [cv2.IMWRITE_JPEG_QUALITY, self.settings["crop_jpeg_quality"]],
            )

            record = {
                "camera_id": cam_id,
                "track_id": track["id"],
                "frame_number": frame_number,
                "timestamp": timestamp,
                "bbox": track["bbox"],
                "crop_path": crop_path,
            }
            self.records.append(record)

            if on_person_detected:
                on_person_detected(record)

    def _save_outputs(self):
        records_path = self.settings["records_json"]
        Path(records_path).parent.mkdir(parents=True, exist_ok=True)
        with open(records_path, "w") as f:
            json.dump(self.records, f, indent=2)
        logger.info(f"✅ Saved {len(self.records)} person records -> {records_path}")

        for cam_id, tracking_data in self.per_camera_tracking.items():
            out_path = f"{self.settings['tracking_dir']}/{cam_id}_tracking.json"
            with open(out_path, "w") as f:
                json.dump(tracking_data, f, indent=2)
            logger.info(f"✅ Saved per-camera tracking -> {out_path}")


def run_multicamera_pipeline(camera_configs=None, device="cpu",
                              conf_threshold=0.35, on_person_detected=None,
                              max_frames=None):
    """Convenience function mirroring the style of run_detection /
    run_tracking / run_reid_pipeline in the rest of the codebase."""
    pipeline = MultiCameraPipeline(
        camera_configs=camera_configs, device=device,
        conf_threshold=conf_threshold,
    )
    return pipeline.run(on_person_detected=on_person_detected, max_frames=max_frames)
