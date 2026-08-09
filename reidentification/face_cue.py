"""
Face-based Re-ID cue.
========================

Body-appearance matching (ResNetReIDBackbone + color/texture cues) struggles
when everyone is dressed identically - there's genuinely little appearance
signal left to distinguish people. Faces don't have that problem: they're
unique regardless of uniform, which is exactly the situation this project's
real demo footage is shot in (clear, front-facing, controlled camera angle -
similar to an attendance-camera setup).

This module is intentionally SEPARATE from MultiCueExtractor's 698-dim
descriptor (reid_main.py) rather than appended into it. Reasons:
  1. Prajna's registration module documents 698-dim as a stable contract
     (IdentityDatabase / export_for_reid()) - growing that vector would
     silently break compatibility with anything already registered against
     it.
  2. A face isn't always visible (person facing away, too far, too blurry) -
     treating it as an optional side-channel that boosts/vetoes a match,
     rather than baking it into a fixed-size vector, degrades gracefully
     instead of injecting zeros that look like a genuine (bad) signal.

Usage (see reid_main.py's ReIDEngine for the actual wiring):
    extractor = FaceCueExtractor()
    face_vec  = extractor.extract(frame, bbox)     # np.ndarray(128,) or None
    sim       = extractor.similarity(vec_a, vec_b)  # 0..1, higher = same person
"""

import logging
import cv2
import numpy as np

from reidentification.face_encoder import get_face_encoder

logger = logging.getLogger(__name__)


class FaceCueExtractor:
    """Extracts a face embedding from the head/face region of a person's
    bounding box, when a face is confidently detected there. The actual
    encoder is pluggable (reidentification.face_encoder): ArcFace 512-dim by
    default, dlib 128-dim as fallback."""

    DIM = None  # set from the active encoder

    # How far down the bounding box to look for a face (head is at the top).
    # Matches roughly the same "face zone" MultiCueExtractor already crops
    # for its color-histogram cue, with a little extra margin.
    HEAD_ZONE_TOP    = 0.0
    HEAD_ZONE_BOTTOM = 0.30

    # Below this face-detector confidence (via the face's location size vs
    # the crop), skip - a tiny/partial face gives a noisy, unreliable
    # embedding that would do more harm than good.
    MIN_FACE_PIXELS = 15   # min face-box side length in the (upsampled) crop

    def __init__(self, upsample_times: int = 2):
        self.enabled = False
        self.upsample_times = upsample_times
        self._errors_logged = 0   # surface the first few extraction errors
                                   # visibly instead of silently swallowing
                                   # them at debug level forever
        try:
            self._encoder = get_face_encoder()
            self.enabled = self._encoder is not None
            type(self).DIM = self._encoder.DIM if self._encoder else None
        except Exception as e:
            logger.warning(f"[WARN] Face encoder unavailable: {e}")

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
        """Returns a face embedding (ArcFace 512-dim or dlib 128-dim), or
        None if no confident face was found in this person's head region."""
        if not self.enabled:
            return None
        try:
            return self._encoder.encode_bbox(frame, bbox)
        except Exception as e:
            if self._errors_logged < 3:
                logger.warning(f"Face extraction error (showing first 3 only): {e}")
                self._errors_logged += 1
            else:
                logger.debug(f"Face extraction error: {e}")
            return None

    @staticmethod
    def similarity(a, b) -> float:
        """0..1 similarity, higher = more likely the same person, using the
        active encoder's metric."""
        if a is None or b is None:
            return None
        try:
            return get_face_encoder().similarity(a, b)
        except Exception:
            return None
