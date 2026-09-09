"""
Live Webcam Recognition (single camera)
========================================

Additive real-time path for the web console: run detection + DeepSORT
tracking + Re-ID against the registered IdentityDatabase on FRAMES arriving
live from a laptop webcam, returning per-track boxes + names so the browser
can draw them over the webcam preview.

This file only ORCHESTRATES the existing per-frame pieces unchanged:

  - detection.detect_module.PersonDetector.detect(frame)
  - tracking.track_module.PersonTracker.update(frame, detections)
  - reidentification.reid_main.ReIDEngine.extract_feature(frame, bbox)
  - reidentification.insight_face.get_shared_extractor()
  - registration.identity_db.IdentityDatabase.match_multimodal(...)

It never edits those modules (or reid_main's clustering internals); the
offline multi-camera pipeline is untouched. Detection settings mirror the
offline camera job (conf 0.6, min_height 50, min_area_ratio 0.001) and the
tracker uses the same run-time defaults, so live outputs stay consistent
with the file-based pipeline in the same 698-dim descriptor space.
"""

import logging
import threading
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class LiveRecognitionSession:
    """Stateful per-webcam session: owns the detector/tracker/reid models and
    keeps per-track feature/face windows so names stay stable across frames.

    One instance per live capture; created by POST /api/live/start and
    released by POST /api/live/{id}/stop. process_frame() is thread-safe.
    """

    def __init__(
        self,
        device="cpu",
        stride=1,
        conf_threshold=0.6,
        imgsz=640,
        min_height=50,
        min_area_ratio=0.001,
        match_window=8,
        name_every=2,
        face_every=1,
        max_faces_per_track=12,
    ):
        self.device = device
        self.stride = max(1, int(stride))
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.min_height = min_height
        self.min_area_ratio = min_area_ratio
        self.match_window = max(1, int(match_window))
        self.name_every = max(1, int(name_every))
        self.face_every = max(1, int(face_every))
        self.max_faces_per_track = max(1, int(max_faces_per_track))

        self._lock = threading.Lock()
        self._models = None
        self._frame_count = 0
        self._last_detections = []
        self._feat_windows = {}   # tid -> deque(698-dim features, capped)
        self._face_counts = {}    # tid -> int (attempts, capped gallery)
        self._track_state = {}    # tid -> {name, similarity, face_sim, cues, faces[]}

    # ── model lifecycle ────────────────────────────────────────────────────

    def _ensure_models(self):
        if self._models is not None:
            return self._models
        with self._lock:
            if self._models is not None:
                return self._models
            from detection.detect_module import PersonDetector
            from tracking.track_module import PersonTracker
            from reidentification.reid_main import ReIDEngine
            from reidentification.insight_face import get_shared_extractor
            from registration.identity_db import IdentityDatabase

            detector = PersonDetector(
                conf_threshold=self.conf_threshold,
                device=self.device,
                min_height=self.min_height,
                min_area_ratio=self.min_area_ratio,
            )
            tracker = PersonTracker(max_age=5, max_cosine_distance=0.5)
            engine = ReIDEngine(device=self.device)
            face_extractor = get_shared_extractor()
            identity_db = IdentityDatabase()
            self._models = {
                "detector": detector,
                "tracker": tracker,
                "reid": engine,
                "faces": face_extractor,
                "db": identity_db,
            }
            logger.info("LiveRecognitionSession: models ready (%d registered person(s))", len(identity_db))
            return self._models

    def warm(self):
        """Load all models now (called by /api/live/start) so the first frame
        the webcam sends is answered quickly instead of paying the full
        YOLO + Re-ID load cost."""
        return self._ensure_models()

    def close(self):
        with self._lock:
            if self._models is not None:
                try:
                    self._models["tracker"].tracker.delete_all_tracks()
                except Exception:
                    pass
                self._models = None
            self._feat_windows.clear()
            self._face_counts.clear()
            self._track_state.clear()
            logger.info("LiveRecognitionSession closed")

    # ── per-frame processing ───────────────────────────────────────────────

    def process_frame(self, frame):
        """Run detection -> tracking -> Re-ID on one webcam frame.

        Args:
            frame: OpenCV BGR frame.

        Returns:
            List of dicts:
                {"track_id": int, "global_id": str|None,
                 "bbox": [x1,y1,x2,y2],
                 "name": str|None, "similarity": float|None,
                 "face_sim": float|None, "cues": [str]}
            bbox is in raw frame-pixel coordinates; the browser scales it to
            the on-screen preview.
        """
        m = self._ensure_models()
        with self._lock:
            self._frame_count += 1
            fc = self._frame_count
            detector = m["detector"]
            tracker = m["tracker"]
            engine = m["reid"]
            faces = m["faces"]
            db = m["db"]

            detect_frame = (fc - 1) % self.stride == 0
            if detect_frame:
                detections = detector.detect(frame, imgsz=self.imgsz)
                self._last_detections = detections
            else:
                detections = self._last_detections

            tracks = tracker.update(frame, detections)

            # Carry over names to new tracks whose centroid is near
            # a previously recognized track's centroid.
            # Uses centroid distance instead of bbox overlap because
            # when a person moves closer/further the bbox size changes
            # dramatically but the centroid stays roughly stable.
            active_tids = {t["id"] for t in tracks}
            for tid in active_tids:
                if self._track_state.get(tid, {}).get("name"):
                    continue  # already named
                bbox = next((t["bbox"] for t in tracks if t["id"] == tid), None)
                if bbox is None:
                    continue
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                for other_tid, other_state in self._track_state.items():
                    if other_tid == tid or other_state.get("name") is None:
                        continue
                    ob = other_state.get("_last_bbox")
                    if not isinstance(ob, (list, tuple)) or len(ob) < 4:
                        continue
                    ox = (ob[0] + ob[2]) / 2
                    oy = (ob[1] + ob[3]) / 2
                    dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                    # threshold: 50% of the current track's diagonal
                    diag = ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5
                    if diag > 0 and dist < 0.5 * diag:
                        self._track_state[tid].update(
                            {k: other_state[k] for k in ("name", "similarity", "face_sim", "cues") if k in other_state}
                        )
                        break

            out = []
            for t in tracks:
                tid = t["id"]
                bbox = t["bbox"]
                state = self._track_state.setdefault(tid, {})
                state["_last_bbox"] = bbox

                if detect_frame:
                    feat = engine.extract_feature(frame, bbox)
                    if feat is None:
                        continue  # cannot describe this track yet (ghost/off-screen)
                    win = self._feat_windows.setdefault(tid, deque(maxlen=self.match_window))
                    win.append(np.asarray(feat, dtype=np.float32))

                    # Face extraction is the slowest per-frame step; run it at
                    # face_every cadence and cap the gallery (offline cap=30,
                    # live keeps it small so match latency stays low).
                    if (fc - 1) % self.face_every == 0:
                        self._face_counts[tid] = self._face_counts.get(tid, 0) + 1
                        if self._face_counts[tid] <= self.max_faces_per_track:
                            f = faces.extract(frame, bbox)
                            if f is not None:
                                state.setdefault("faces", []).append(np.asarray(f, dtype=np.float32))

                    if (fc - 1) % self.name_every == 0:
                        match = self._resolve(db, win, state.get("faces", []))
                        if match["name"]:
                            state.update(match)      # a confirmed name sticks
                        else:
                            if "name" not in state:
                                state.update(match)  # still unknown: keep it explicit

                out.append(
                    {
                        "track_id": tid,
                        "global_id": state.get("name"),
                        "bbox": bbox,
                        "name": state.get("name"),
                        "similarity": state.get("similarity"),
                        "face_sim": state.get("face_sim"),
                        "cues": state.get("cues") or [],
                    }
                )

            self._prune({t["id"] for t in tracks})
            return out

    def _resolve(self, db, win, face_list):
        """Match a track's rolling mean appearance (+ faces) against the
        registered database. Mirrors the offline name resolution: same
        match_multimodal call, same thresholds."""
        if not win or not len(db):
            return {"name": None, "similarity": None, "face_sim": None, "cues": []}
        mean_feat = np.mean(list(win), axis=0)
        try:
            matches = db.match_multimodal(mean_feat, list(face_list), top_k=1, threshold=None)
        except Exception:
            logger.exception("Live identity match failed — leaving track unresolved")
            matches = []
        if not matches:
            return {"name": None, "similarity": None, "face_sim": None, "cues": []}
        m = matches[0]
        return {
            "name": m["name"],
            "similarity": m["score"],
            "face_sim": m["face_sim"],
            "cues": m["cues"],
        }

    def _prune(self, active_tids):
        """Drop windows/faces/names for tracks no longer on screen so a long
        live session can't grow unbounded state."""
        gone = [tid for tid in self._track_state if tid not in active_tids]
        for tid in gone:
            self._feat_windows.pop(tid, None)
            self._face_counts.pop(tid, None)
            self._track_state.pop(tid, None)