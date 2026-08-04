"""
Face-based Re-ID cue (ONNX backend).
====================================

Body-appearance matching (ResNetReIDBackbone + color/texture cues) struggles
when everyone is dressed identically — there's genuinely little appearance
signal left to distinguish people. Faces don't have that problem: they're
unique regardless of uniform, which is exactly the situation this project's
real demo footage is shot in (clear, front-facing, controlled camera angle —
similar to an attendance-camera setup).

This module is intentionally SEPARATE from MultiCueExtractor's 698-dim
descriptor (reid_main.py) rather than appended into it. Reasons:
  1. Prajna's registration module documents 698-dim as a stable contract
     (IdentityDatabase / export_for_reid()) — growing that vector would
     silently break compatibility with anything already registered against
     it.
  2. A face isn't always visible (person facing away, too far, too blurry) —
     treating it as an optional side-channel that boosts/vetoes a match,
     rather than baking it into a fixed-size vector, degrades gracefully
     instead of injecting zeros that look like a genuine (bad) signal.

Backend (why not dlib/face_recognition):
  face_recognition depends on dlib, which ships NO Windows wheels on PyPI
  for any Python version and needs MSVC + CMake to build from source — so it
  silently disables itself on most student laptops. This module instead uses
  two OpenCV-Zoo ONNX models (loaded lazily, no compilation):
    - YuNet        : face detector  -> face bbox + landmarks in the head zone
    - SFace (128-d): identity embedding for the aligned face
  SFace is also more accurate than dlib's ResNet-34 (LFW ~99.6% vs ~99.4%).
  The public API is unchanged (extract -> 128-dim or None, similarity -> 0..1)
  and the output stays a cosine score, which matches SFace's own same-person
  threshold of 0.363 and the callers' existing FACE_MIN_SIMILARITY-style bars.

Usage (see reid_main.py's ReIDEngine for the actual wiring):
    extractor = FaceCueExtractor()
    face_vec  = extractor.extract(frame, bbox)     # np.ndarray(128,) or None
    sim       = extractor.similarity(vec_a, vec_b)  # 0..1, higher = same person
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent / "weights"
_DETECTOR_PATH   = _MODELS_DIR / "face_detection_yunet_2023mar.onnx"
_RECOGNIZER_PATH = _MODELS_DIR / "face_recognition_sface_2021dec.onnx"


def _models_available() -> bool:
    return _DETECTOR_PATH.exists() and _RECOGNIZER_PATH.exists()


class FaceCueExtractor:
    """Extracts a 128-dim face embedding from the head/face region of a
    person's bounding box, when a confident face is detected there."""

    DIM = 128

    # The old head-zone trick (top 30% of the box) assumed full-body
    # detections where the face is small; registration photos and tight
    # person crops have large faces that get sliced in half. We now scan
    # the whole box and let YuNet pick the largest face — this zone crop is
    # kept only for reference / fallback.
    HEAD_ZONE_TOP    = 0.0
    HEAD_ZONE_BOTTOM = 1.0

    # Below this face-detector confidence (via the face's location size vs
    # the crop), skip — a tiny/partial face gives a noisy, unreliable
    # embedding that would do more harm than good.
    MIN_FACE_PIXELS = 15   # min face-box side length (in crop pixels)

    def __init__(self, upsample_times: int = 2):
        # Kept for API compatibility with the old face_recognition backend;
        # YuNet resizes its input internally, so upsampling is unnecessary.
        self.upsample_times = upsample_times
        self._detector  = None
        self._recognizer = None
        self._errors_logged = 0   # surface the first few extraction errors
                                   # visibly instead of silently swallowing
                                   # them at debug level forever
        self.enabled = _models_available()
        if self.enabled:
            try:
                self._detector = cv2.FaceDetectorYN_create(
                    str(_DETECTOR_PATH),
                    "",
                    (320, 320),
                    score_threshold=0.8,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                self._recognizer = cv2.FaceRecognizerSF.create(
                    str(_RECOGNIZER_PATH),
                    "",
                    backend_id=0,
                    target_id=0,
                )
            except Exception as exc:
                logger.warning(f"⚠️  Face cue disabled — failed to load ONNX models: {exc}")
                self.enabled = False
        if not self.enabled:
            logger.warning(
                "⚠️  Face-based Re-ID cue disabled (ONNX models not found in "
                f"{_MODELS_DIR}) — falling back to body-appearance-only matching."
            )

    def _head_crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        h = y2 - y1
        top = y1 + int(h * self.HEAD_ZONE_TOP)
        bot = min(y2, y1 + int(h * self.HEAD_ZONE_BOTTOM) + 1)
        crop = frame[top:bot, x1:x2]
        return crop if crop.size > 0 else None

    def extract(self, frame, bbox):
        """Returns a 128-dim face embedding, or None if no confident face
        was found in this person's bounding box region."""
        result = self.extract_with_score(frame, bbox)
        return result[0] if result is not None else None

    def extract_with_score(self, frame, bbox):
        """Returns (embedding, detector_score) or None. detector_score is the
        YuNet confidence of the chosen face, so callers can pick the best
        face across a track (the engine itself keeps only the last-seen
        face, which is often blurrier than the best one)."""
        result = self.extract_with_box(frame, bbox)
        return (result[0], result[1]) if result is not None else None

    def extract_with_box(self, frame, bbox):
        """Returns (embedding, detector_score, face_bbox) or None.

        face_bbox is (x, y, w, h) in the same coordinate space as the input
        person bbox, so callers can judge whether the detected face is large
        enough to be trustworthy (a tiny/partial face should not override a
        good body match).
        """
        if not self.enabled or self._detector is None or self._recognizer is None:
            return None
        crop = self._head_crop(frame, bbox)
        if crop is None or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        try:
            # Head crops are usually small, so upscale before detecting —
            # the same role face_recognition's upsample_times played. YuNet
            # needs its input size preset to the (upscaled) crop size.
            scale = 2
            big = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                             interpolation=cv2.INTER_LINEAR)
            self._detector.setInputSize((big.shape[1], big.shape[0]))
            _, faces = self._detector.detect(big)
            if faces is None or len(faces) == 0:
                return None
            # pick the largest face found (most likely the actual subject,
            # not a smaller face bleeding in from a neighbouring crop)
            def _area(f):
                return float(f[2]) * float(f[3])
            best = faces[np.argmax([_area(f) for f in faces])]
            if min(float(best[2]), float(best[3])) < self.MIN_FACE_PIXELS * scale:
                return None   # face too small/partial to trust

            score = float(best[-1])
            aligned = self._recognizer.alignCrop(big, best.astype(np.float32))
            feat = self._recognizer.feature(aligned)
            # Map the face box back to the input person-bbox space.
            x1, y1, x2, y2 = map(int, bbox)
            h = y2 - y1
            crop_top = y1 + int(h * self.HEAD_ZONE_TOP)
            crop_left = x1
            fx = crop_left + float(best[0]) / scale
            fy = crop_top + float(best[1]) / scale
            fbox = (fx, fy, float(best[2]) / scale, float(best[3]) / scale)
            return (np.asarray(feat, dtype=np.float32).reshape(-1), score, fbox)
        except Exception as e:
            if self._errors_logged < 3:
                logger.warning(f"⚠️  Face extraction error (showing first 3 only): {e}")
                self._errors_logged += 1
            else:
                logger.debug(f"Face extraction error: {e}")
            return None

    @staticmethod
    def similarity(a, b) -> float | None:
        """0..1 similarity, higher = more likely the same person. SFace
        embeddings are L2-normalised and compared by cosine similarity; the
        library's own same-person cutoff is ~0.363, which sits between the
        callers' veto (0.35) and merge (0.45) thresholds."""
        if a is None or b is None:
            return None
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.clip(float(np.dot(a, b) / (na * nb)), 0.0, 1.0))
