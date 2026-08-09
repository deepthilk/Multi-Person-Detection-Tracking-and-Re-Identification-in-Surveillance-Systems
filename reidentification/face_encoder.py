"""
Pluggable face encoder backend.
================================

The face stack (face DB, per-track face sampling, wanted-person filter, name
resolution) must always compare embeddings produced by the SAME encoder:
a 512-dim ArcFace vector compared against a 128-dim dlib vector would be
nonsense. This module is the single place that owns:

  * which backend is active (ArcFace preferred, dlib fallback),
  * how an embedding is produced from a person's bounding box / an image,
  * what "distance" and "similarity" mean for that backend,
  * what the confident same-person distance cutoff is.

Backends
--------
arcface   - insightface "buffalo_l" (SCRFD detection + 106pt landmarks +
            w600k_r50 ArcFace recognizer, 512-dim). Robust across different
            photos/lighting; much better than dlib at matching a registration
            photo taken at a different time/angle to live footage.
dlib      - face_recognition's 128-dim embeddings (the historical backend).

Every consumer (FaceCueExtractor, FaceIdentityDB, face name resolution)
calls get_face_encoder() instead of importing face_recognition directly, so
switching backends just flips FACE_ENCODER_BACKEND and rebuilding the face DB.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Which backend to use: "arcface" (preferred) or "dlib" (fallback).
FACE_ENCODER_BACKEND = "arcface"

# Confident same-person distance cutoff per backend. Lower = stricter.
#   dlib    : Euclidean distance, 0.40 (footage-calibrated, see face_db.py).
#   arcface : cosine distance 1 - cos(a, b); 0.40 => cos >= 0.60. Final value
#             is calibrated on this project's footage in face_db rebuild logs.
_CONFIRM_DISTANCE = {"arcface": 0.45, "dlib": 0.40}

# Head zone of a person bounding box where the face lives (fraction of height).
HEAD_ZONE_TOP = 0.0
HEAD_ZONE_BOTTOM = 0.30

# Min detected face side (px) we are willing to embed.
MIN_FACE_PIXELS = 15


class BaseFaceEncoder:
    """Interface all backends implement. Embeddings are np.ndarray(float32)."""

    name = "base"
    DIM = None

    def encode_image_faces(self, image, max_faces=1, min_face_px=MIN_FACE_PIXELS):
        """Detect faces in a full BGR image. Returns list of embeddings,
        largest face first. Empty list if no confident face found."""
        raise NotImplementedError

    def encode_bbox(self, frame, bbox):
        """Encode the best face found in the head zone of a person's bounding
        box. Returns a (embedding,) np.ndarray or None."""
        raise NotImplementedError

    def distance(self, a, b):
        """Distance between two embeddings. Lower = more likely same person."""
        raise NotImplementedError

    def similarity(self, a, b):
        """0..1 similarity, higher = more likely same person."""
        raise NotImplementedError

    def distance_to_similarity(self, dist):
        raise NotImplementedError

    def confirm_distance(self):
        return _CONFIRM_DISTANCE.get(self.name, 0.4)


class DlibFaceEncoder(BaseFaceEncoder):
    name = "dlib"
    DIM = 128

    def __init__(self):
        import face_recognition
        self._fr = face_recognition

    def encode_image_faces(self, image, max_faces=1, min_face_px=MIN_FACE_PIXELS):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        locations = self._fr.face_locations(rgb, number_of_times_to_upsample=1)
        if not locations:
            return []
        locations = sorted(
            locations,
            key=lambda l: (l[1] - l[3]) * (l[2] - l[0]),
            reverse=True,
        )
        encodings = []
        for top, right, bottom, left in locations:
            if min(right - left, bottom - top) < min_face_px:
                continue
            encs = self._fr.face_encodings(
                rgb, known_face_locations=[(top, right, bottom, left)])
            if encs:
                encodings.append(np.asarray(encs[0], dtype=np.float32))
            if len(encodings) >= max_faces:
                break
        return encodings

    def encode_bbox(self, frame, bbox):
        x1, y1, x2, y2 = _clamped_bbox(frame, bbox)
        if x2 <= x1 or y2 <= y1:
            return None
        h = y2 - y1
        top = y1 + int(h * HEAD_ZONE_TOP)
        bot = min(y2, y1 + int(h * HEAD_ZONE_BOTTOM) + 1)
        crop = frame[top:bot, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        locations = self._fr.face_locations(rgb, number_of_times_to_upsample=2)
        if not locations:
            return None

        def _area(loc):
            t, r, b, l = loc
            return max(0, r - l) * max(0, b - t)

        best = max(locations, key=_area)
        top, right, bottom, left = best
        if min(right - left, bottom - top) < MIN_FACE_PIXELS:
            return None
        encs = self._fr.face_encodings(rgb, known_face_locations=[best])
        return np.asarray(encs[0], dtype=np.float32) if encs else None

    def distance(self, a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return float(np.linalg.norm(a - b))

    def similarity(self, a, b):
        d = self.distance(a, b)
        return self.distance_to_similarity(d)

    def distance_to_similarity(self, dist):
        return float(np.clip(1.0 - (dist / 0.9), 0.0, 1.0))


class ArcFaceFaceEncoder(BaseFaceEncoder):
    name = "arcface"
    DIM = 512

    def __init__(self, app=None):
        self._app = app or self._build_app()

    @staticmethod
    def _build_app():
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        logger.info("ArcFace encoder ready (buffalo_l: SCRFD + w600k_r50, 512-dim)")
        return app

    def _embed(self, face):
        emb = np.asarray(face.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else None

    def encode_image_faces(self, image, max_faces=1, min_face_px=MIN_FACE_PIXELS):
        faces = self._app.get(image)
        if not faces:
            return []
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                       reverse=True)
        encodings = []
        for f in faces:
            if min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) < min_face_px:
                continue
            emb = self._embed(f)
            if emb is not None:
                encodings.append(emb)
            if len(encodings) >= max_faces:
                break
        return encodings

    def encode_bbox(self, frame, bbox):
        x1, y1, x2, y2 = _clamped_bbox(frame, bbox)
        if x2 <= x1 or y2 <= y1:
            return None
        h = y2 - y1
        top = y1 + int(h * HEAD_ZONE_TOP)
        bot = min(y2, y1 + int(h * HEAD_ZONE_BOTTOM) + 1)
        crop = frame[top:bot, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        faces = self._app.get(crop)
        if faces:
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if min(best.bbox[2] - best.bbox[0], best.bbox[3] - best.bbox[1]) >= MIN_FACE_PIXELS:
                return self._embed(best)
        # fallback: run detection on the whole frame and keep faces whose
        # center lands inside the head zone (robust to small head crops).
        faces = self._app.get(frame)
        if not faces:
            return None
        best = None
        best_area = 0
        for f in faces:
            cx = (f.bbox[0] + f.bbox[2]) / 2
            cy = (f.bbox[1] + f.bbox[3]) / 2
            if not (x1 <= cx <= x2 and top <= cy <= bot):
                continue
            area = (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            if area > best_area:
                best_area = area
                best = f
        if best is None:
            return None
        return self._embed(best)

    def distance(self, a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 2.0
        return float(1.0 - (a @ b) / (na * nb))

    def similarity(self, a, b):
        return float(1.0 - self.distance(a, b))

    def distance_to_similarity(self, dist):
        return float(np.clip(1.0 - dist, 0.0, 1.0))


def _clamped_bbox(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    return x1, y1, x2, y2


_ACTIVE_ENCODER = None


def get_face_encoder():
    """The active FaceEncoder instance (cached singleton). Falls back to dlib
    when arcface is requested but insightface / the model is unavailable, and
    raises a clear error only if no backend can be built at all."""
    global _ACTIVE_ENCODER
    if _ACTIVE_ENCODER is not None:
        return _ACTIVE_ENCODER
    requested = FACE_ENCODER_BACKEND
    if requested == "arcface":
        try:
            _ACTIVE_ENCODER = ArcFaceFaceEncoder()
            return _ACTIVE_ENCODER
        except Exception as e:
            logger.warning(
                f"[WARN] ArcFace encoder unavailable ({e}); falling back to dlib")
    try:
        _ACTIVE_ENCODER = DlibFaceEncoder()
    except Exception as e:
        raise RuntimeError(
            "No face encoder available: neither insightface/ArcFace nor "
            f"face_recognition/dlib could be loaded ({e})") from e
    return _ACTIVE_ENCODER


def face_distance(query_encodings, candidate):
    """Minimum encoder distance between a list of query encodings and one
    candidate encoding. None if no query encodings."""
    if not query_encodings or candidate is None:
        return None
    enc = get_face_encoder()
    return float(min(enc.distance(q, candidate) for q in query_encodings))


def distance_to_similarity(dist):
    """Encoder-consistent 0..1 similarity for a distance."""
    if dist is None:
        return None
    return get_face_encoder().distance_to_similarity(dist)


def confirm_distance():
    """Confident same-person distance cutoff for the active encoder."""
    return get_face_encoder().confirm_distance()


def reset_encoder():
    """Drop the cached encoder (tests / after backend switch)."""
    global _ACTIVE_ENCODER
    _ACTIVE_ENCODER = None
