"""
InsightFace-backed face extractor for registration & video search.
===================================================================

`reidentification/face_cue.py`'s FaceCueExtractor relies on dlib via the
`face_recognition` package, which cannot be built on this machine (no
cmake/MSVC toolchain).  So the registration gallery and the video-search
face cue use a DIFFERENT extractor here, backed by InsightFace (ArcFace
`w600k_mbf` ONNX model from the `buffalo_s` model pack):

  - extract(frame, bbox) -> np.ndarray(512,) | None
    Detects the largest face inside the person's HEAD region (top of the
    bounding box), aligns + embeds it.  Vectors are L2-normalised, so the
    same feature space is used on BOTH the registration side and the
    search side — cosine similarity is directly comparable.

  - similarity(a, b) -> float (0..1 cosine)

The live Re-ID engine (reid_main.py / track_cluster.py) keeps using
FaceCueExtractor, which is simply disabled when dlib is missing — so this
module never changes validated live-tracking behaviour.
"""

import logging

import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False
    logger.warning(
        "⚠️  'insightface' not installed — face galleries/search disabled. "
        "Install with: pip install insightface onnxruntime"
    )


class InsightFaceExtractor:
    """Extracts a 512-dim ArcFace embedding from the head region of a
    person's bounding box, when a confident face is detected there."""

    DIM = 512

    # How far down the bounding box to look for a face (head is at the top).
    HEAD_ZONE_TOP    = 0.0
    HEAD_ZONE_BOTTOM = 0.40

    # Below this face-box side length (px in the ORIGINAL frame), skip —
    # a tiny/partial face gives a noisy embedding.
    MIN_FACE_PIXELS = 15

    def __init__(self, det_size=(640, 640), max_batch=1):
        self.enabled = _INSIGHTFACE_AVAILABLE
        self._app = None
        self._det_size = det_size
        self._max_batch = max_batch
        # FaceAnalysis and the detector's input_size are shared mutable state:
        # guard calls so concurrent camera jobs can share one extractor safely.
        self._lock = threading.Lock()

    def _get_app(self):
        if self._app is None:
            if not self.enabled:
                return None
            self._app = FaceAnalysis(
                name="buffalo_s",
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            self._app.prepare(ctx_id=-1, det_size=self._det_size)
            logger.info("InsightFace extractor ready (buffalo_s / ArcFace w600k_mbf)")
        return self._app

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

    def extract(self, frame, bbox, use_head_region=True):
        """Returns a 512-dim L2-normalised face embedding, or None if no
        confident face was found.

        - use_head_region=True  (video): search only the top strip of the
          person's bbox — a tracked bbox wraps the whole body, and the head
          is at its top.
        - use_head_region=False (registration photos): search the ENTIRE
          image. A registered photo IS the person crop, so a top-40% strip
          can cut off or miss a face that sits lower in the frame.
        """
        if use_head_region:
            crop = self._head_crop(frame, bbox)
            if crop is None or crop.shape[0] < 20 or crop.shape[1] < 20:
                return None
            return self._best_face_embedding(crop)
        return self._best_face_embedding(frame)

    def _best_face_embedding(self, image):
        with self._lock:
            return self._best_face_embedding_unlocked(image)

    def _best_face_embedding_unlocked(self, image):
        app = self._get_app()
        if app is None:
            return None
        try:
            faces = app.get(image, max_num=1)
            if not faces:
                # small faces missed at det_size 640 — retry on a finer grid
                old = app.det_model.input_size
                app.det_model.input_size = (320, 320)
                try:
                    faces = app.get(image, max_num=1)
                finally:
                    app.det_model.input_size = old
            if not faces:
                return None
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if min(best.bbox[2] - best.bbox[0], best.bbox[3] - best.bbox[1]) < self.MIN_FACE_PIXELS:
                return None
            emb = best.normed_embedding
            if emb is None:
                return None
            return np.asarray(emb, dtype=np.float32)
        except Exception as e:
            logger.debug(f"InsightFace extraction error: {e}")
            return None

    @staticmethod
    def similarity(a, b) -> float:
        """0..1 cosine similarity between two ArcFace embeddings (both are
        L2-normalised, so this is simply the clipped dot product)."""
        if a is None or b is None:
            return None
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


_shared_extractor = None
_shared_extractor_lock = threading.Lock()


def get_shared_extractor():
    """Process-wide singleton InsightFaceExtractor.

    Loading the buffalo_s ONNX models costs several seconds per instance, and
    a single video job otherwise creates TWO extractors (the Re-ID loop plus
    the face-verification pass). Sharing one instance across the process loads
    the models once and removes that fixed cost from every job. The creation
    lock keeps concurrent camera jobs from double-loading the models.
    """
    global _shared_extractor
    if _shared_extractor is None:
        with _shared_extractor_lock:
            if _shared_extractor is None:
                _shared_extractor = InsightFaceExtractor()
    return _shared_extractor
