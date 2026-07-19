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
