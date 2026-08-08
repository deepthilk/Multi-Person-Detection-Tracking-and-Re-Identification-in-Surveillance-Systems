"""
Face-based Re-ID cue — Hybrid: YuNet detection + ArcFace recognition.
=====================================================================

Detection: YuNet (OpenCV FaceDetectorYN) — reliable, battle-tested,
handles any input size internally, provides 5-landmark face alignment.

Recognition: ArcFace R100 (ONNX) — 512-dim embeddings, LFW 99.83%,
significantly more discriminative than SFace (128-dim, ~99.6%).

The hybrid combines YuNet's rock-solid detection + alignment with
ArcFace's superior recognition accuracy.

Public API is unchanged: extract() -> 512-dim or None, similarity() -> 0..1.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent / "weights"
_DETECTOR_PATH   = _MODELS_DIR / "face_detection_yunet_2023mar.onnx"
_ARCFACE_PATH    = _MODELS_DIR / "arcface_r100.onnx"


def _models_available() -> bool:
    return _DETECTOR_PATH.exists() and _ARCFACE_PATH.exists()


class FaceCueExtractor:
    """Extracts a 512-dim face embedding from the head/face region of a
    person's bounding box, using YuNet detection + ArcFace recognition."""

    DIM = 512

    HEAD_ZONE_TOP    = 0.0
    HEAD_ZONE_BOTTOM = 1.0

    MIN_FACE_PIXELS = 15

    def __init__(self):
        self._detector = None
        self._arcface_session = None
        self._arcface_input = None
        self._errors_logged = 0
        self.enabled = _models_available()
        if self.enabled:
            try:
                # YuNet detector via OpenCV
                self._detector = cv2.FaceDetectorYN_create(
                    str(_DETECTOR_PATH),
                    "",
                    (320, 320),
                    score_threshold=0.8,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                # ArcFace recognizer via ONNX Runtime
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                self._arcface_session = ort.InferenceSession(
                    str(_ARCFACE_PATH), opts, providers=["CPUExecutionProvider"]
                )
                self._arcface_input = self._arcface_session.get_inputs()[0].name
            except Exception as exc:
                logger.warning(f"Face cue disabled - failed to load models: {exc}")
                self.enabled = False
        if not self.enabled:
            logger.warning(
                "Face-based Re-ID cue disabled (models not found in "
                f"{_MODELS_DIR}) - falling back to body-appearance-only matching."
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

    def _arcface_embed(self, face_img):
        """Extract 512-dim ArcFace embedding from an aligned face crop.

        Input: BGR image (any size from YuNet's bounding-box crop).
        Preprocessing: resize to 112x112, BGR->RGB, normalize to [-1,1],
        feed as NHWC (model's expected format).
        """
        if self._arcface_session is None:
            return None
        face = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_LINEAR)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 127.5 - 1.0
        face = np.expand_dims(face, 0)  # NHWC [1, 112, 112, 3]
        embedding = self._arcface_session.run(None, {self._arcface_input: face})[0][0]
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return None
        return (embedding / norm).astype(np.float32)

    def extract(self, frame, bbox):
        """Returns a 512-dim face embedding, or None if no confident face
        was found in this person's bounding box region."""
        result = self.extract_with_score(frame, bbox)
        return result[0] if result is not None else None

    def extract_with_score(self, frame, bbox):
        """Returns (embedding, detector_score) or None."""
        result = self.extract_with_box(frame, bbox)
        return (result[0], result[1]) if result is not None else None

    def extract_with_box(self, frame, bbox):
        """Returns (embedding, detector_score, face_bbox) or None.

        face_bbox is (x, y, w, h) in the same coordinate space as the input
        person bbox.
        """
        if not self.enabled or self._detector is None or self._arcface_session is None:
            return None
        crop = self._head_crop(frame, bbox)
        if crop is None or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        try:
            # Upscale small crops for better detection (same as old working code)
            scale = 2
            big = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                             interpolation=cv2.INTER_LINEAR)
            self._detector.setInputSize((big.shape[1], big.shape[0]))
            _, faces = self._detector.detect(big)
            if faces is None or len(faces) == 0:
                return None

            # Pick the largest face
            def _area(f):
                return float(f[2]) * float(f[3])
            best = faces[np.argmax([_area(f) for f in faces])]
            if min(float(best[2]), float(best[3])) < self.MIN_FACE_PIXELS * scale:
                return None

            score = float(best[-1])

            # Crop the aligned face from the upscaled image
            ix1 = max(0, int(best[0]))
            iy1 = max(0, int(best[1]))
            ix2 = min(big.shape[1], int(best[0] + best[2]))
            iy2 = min(big.shape[0], int(best[1] + best[3]))
            face_crop = big[iy1:iy2, ix1:ix2]
            if face_crop.size == 0:
                return None

            # Extract ArcFace embedding from the face crop
            feat = self._arcface_embed(face_crop)
            if feat is None:
                return None

            # Map face box back to original person-bbox space
            x1, y1, x2, y2 = map(int, bbox)
            h = y2 - y1
            crop_top = y1 + int(h * self.HEAD_ZONE_TOP)
            crop_left = x1
            fx = crop_left + float(best[0]) / scale
            fy = crop_top + float(best[1]) / scale
            fbox = (fx, fy, float(best[2]) / scale, float(best[3]) / scale)
            return (feat, score, fbox)
        except Exception as e:
            if self._errors_logged < 3:
                logger.warning(f"Face extraction error (showing first 3 only): {e}")
                self._errors_logged += 1
            else:
                logger.debug(f"Face extraction error: {e}")
            return None

    @staticmethod
    def similarity(a, b) -> float | None:
        """0..1 similarity, higher = more likely the same person.
        ArcFace embeddings are L2-normalised and compared by cosine similarity.
        Returns 0.0 if dimensions mismatch (legacy SFace vs ArcFace)."""
        if a is None or b is None:
            return None
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if a.shape[0] != b.shape[0]:
            return 0.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.clip(float(np.dot(a, b) / (na * nb)), 0.0, 1.0))


_face_extractor_cache = None


def get_cached_face_extractor() -> "FaceCueExtractor":
    global _face_extractor_cache
    if _face_extractor_cache is None:
        _face_extractor_cache = FaceCueExtractor()
    return _face_extractor_cache
