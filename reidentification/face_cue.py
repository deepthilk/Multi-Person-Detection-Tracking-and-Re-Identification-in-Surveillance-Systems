"""
Face-based Re-ID cue.
========================

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

Usage (see reid_main.py's ReIDEngine for the actual wiring):
    extractor = FaceCueExtractor()
    face_vec  = extractor.extract(frame, bbox)     # np.ndarray(128,) or None
    sim       = extractor.similarity(vec_a, vec_b)  # 0..1, higher = same person
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import face_recognition
    _FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    _FACE_RECOGNITION_AVAILABLE = False
    logger.warning(
        "⚠️  'face_recognition' not installed — face-based Re-ID cue disabled, "
        "falling back to body-appearance-only matching. "
        "Install with: pip install face_recognition"
    )


class FaceCueExtractor:
    """Extracts a 128-dim face embedding from the head/face region of a
    person's bounding box, when a face is confidently detected there."""

    DIM = 128

    # How far down the bounding box to look for a face (head is at the top).
    # Matches roughly the same "face zone" MultiCueExtractor already crops
    # for its color-histogram cue, with a little extra margin.
    HEAD_ZONE_TOP    = 0.0
    HEAD_ZONE_BOTTOM = 0.30

    # Below this face-detector confidence (via the face's location size vs
    # the crop), skip — a tiny/partial face gives a noisy, unreliable
    # embedding that would do more harm than good.
    MIN_FACE_PIXELS = 20   # min face-box side length in the (upsampled) crop

    def __init__(self, upsample_times: int = 1):
        self.enabled = _FACE_RECOGNITION_AVAILABLE
        self.upsample_times = upsample_times

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
        was found in this person's head region."""
        if not self.enabled:
            return None
        crop = self._head_crop(frame, bbox)
        if crop is None or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        try:
            # face_recognition expects RGB
            rgb = crop[:, :, ::-1]
            locations = face_recognition.face_locations(
                rgb, number_of_times_to_upsample=self.upsample_times
            )
            if not locations:
                return None
            # pick the largest face found (most likely the actual subject,
            # not a smaller face bleeding in from a neighbouring crop)
            def _area(loc):
                top, right, bottom, left = loc
                return max(0, right - left) * max(0, bottom - top)
            best = max(locations, key=_area)
            top, right, bottom, left = best
            if min(right - left, bottom - top) < self.MIN_FACE_PIXELS:
                return None   # face too small/partial to trust

            encodings = face_recognition.face_encodings(rgb, known_face_locations=[best])
            if not encodings:
                return None
            return np.asarray(encodings[0], dtype=np.float32)
        except Exception as e:
            logger.debug(f"Face extraction error: {e}")
            return None

    @staticmethod
    def similarity(a, b) -> float:
        """0..1 similarity, higher = more likely the same person. face_recognition
        embeddings are compared by Euclidean distance; ~0.6 is the library's own
        conventional same-person cutoff, so we map distance -> similarity around
        that scale rather than assuming a 0..1 cosine-style range."""
        if a is None or b is None:
            return None
        dist = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
        sim = 1.0 - (dist / 0.9)   # ~0 distance -> 1.0, ~0.9 distance -> 0.0
        return float(np.clip(sim, 0.0, 1.0))
