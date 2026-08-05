"""
Embedding generation for the Registration module.

IMPORTANT DESIGN DECISION
--------------------------
Registered-person embeddings MUST live in the same feature space as the
embeddings produced during live Re-ID matching, or cosine-similarity
comparisons between "known person" and "person seen on camera" would be
meaningless.

So instead of inventing a second, incompatible embedding model, this module
imports and reuses Deepthi's existing `ReIDEngine` from
`reidentification/reid_main.py` — it is only ever CALLED, never edited.
This mirrors exactly how Lekha's `multicamera` module imports
`PersonDetector` / `PersonTracker` without touching them.

If Deepthi later swaps the backbone (e.g. real OSNet instead of the
ResNet-50 fallback), nothing here needs to change — `extract_feature()`
still returns whatever the current descriptor is.
"""

import logging

import cv2
import numpy as np

from registration.db_config import EMBEDDING_SETTINGS

logger = logging.getLogger(__name__)

_engine = None  # lazily created, shared across calls in one process


def _get_engine():
    """Create (once) and return the shared ReIDEngine instance."""
    global _engine
    if _engine is None:
        # Imported lazily so the registration module can be imported /
        # unit-tested even in environments where torch/ultralytics aren't
        # fully set up yet.
        from reidentification.reid_main import ReIDEngine

        device = EMBEDDING_SETTINGS["device"]
        logger.info(f"Loading shared Re-ID backbone for registration (device={device})...")
        _engine = ReIDEngine(device=device)
    return _engine


def embed_image(image_path_or_array) -> np.ndarray:
    """
    Produce a single 698-dim descriptor for a person image.

    Accepts either a file path (str) or an already-loaded BGR image
    (numpy array), which is convenient both for CLI registration from
    disk and for future use with images already in memory (e.g. an
    upload from the dashboard).

    Returns None if the image can't be read or is too small.
    """
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
        if image is None:
            logger.warning(f"Could not read image: {image_path_or_array}")
            return None
    else:
        image = image_path_or_array

    h, w = image.shape[:2]
    min_size = EMBEDDING_SETTINGS["min_image_size"]
    if h < min_size or w < min_size:
        logger.warning(f"Image too small ({w}x{h}), skipping")
        return None

    engine = _get_engine()

    # The registered photo IS the person crop (no detector needed here),
    # so the "bbox" passed to the shared extractor is simply the full image.
    bbox = [0, 0, w, h]
    feature = engine.extract_feature(image, bbox)

    if feature is None:
        logger.warning("Feature extraction returned None for this image")
    return feature


def embed_images(image_paths) -> list:
    """Embed a list of images, silently skipping any that fail."""
    embeddings = []
    for path in image_paths:
        feat = embed_image(path)
        if feat is not None:
            embeddings.append(feat)
    return embeddings


_face_extractor = None


def _get_face_extractor(upsample_times: int):
    """Lazily build (and cache) the face extractor used to build registration
    face galleries. Separate from the engine's internal instance so it never
    changes live Re-ID face-cue behaviour. Backed by InsightFace (ArcFace
    w600k_mbf) — `upsample_times` is accepted for backward compatibility but
    ignored (the insightface detector sizes the input via det_size)."""
    global _face_extractor
    if _face_extractor is None:
        from reidentification.insight_face import InsightFaceExtractor
        _face_extractor = InsightFaceExtractor()
    return _face_extractor


def embed_image_with_face(image_path_or_array, face_upsample: int = 3) -> dict:
    """
    Produce BOTH the 698-dim body-appearance descriptor and, when a confident
    face is visible, a 512-dim ArcFace face embedding for a person image.

    Registration stores the face vectors as a per-person gallery so the search
    side can match by FACE in addition to body appearance ("register a person
    with many faces, then find them in the video"). The face extractor is the
    same InsightFaceExtractor the search side uses, so the vectors live in the
    exact same space as faces collected from the video.

    `face_upsample` is accepted for backward compatibility but ignored
    (insightface sizes input via det_size).

    Returns {"appearance": np.ndarray(698,) | None, "face": np.ndarray(512,) | None},
    or None if the image can't be read / is too small.
    """
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
        if image is None:
            logger.warning(f"Could not read image: {image_path_or_array}")
            return None
    else:
        image = image_path_or_array

    h, w = image.shape[:2]
    min_size = EMBEDDING_SETTINGS["min_image_size"]
    if h < min_size or w < min_size:
        logger.warning(f"Image too small ({w}x{h}), skipping")
        return None

    engine = _get_engine()
    bbox = [0, 0, w, h]

    appearance = engine.extract_feature(image, bbox)
    face = None
    if appearance is not None:
        face = _get_face_extractor(face_upsample).extract(image, bbox,
                                                          use_head_region=False)
    return {"appearance": appearance, "face": face}


def embed_images_with_face(image_paths, face_upsample: int = 3) -> list:
    """Embed a list of images for registration, silently skipping any whose
    body-appearance descriptor fails (a missing face is NOT a failure — it's
    just stored without a face vector)."""
    out = []
    for path in image_paths:
        res = embed_image_with_face(path, face_upsample=face_upsample)
        if res is not None and res["appearance"] is not None:
            out.append(res)
    return out
