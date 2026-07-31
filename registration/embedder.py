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

DOMAIN-SHIFT FIX (Phase 2, Prajna)
-----------------------------------
The Re-ID model was fine-tuned on Market-1501 (surveillance-camera crops).
When registration uses a high-resolution professional photo (different
lighting, no compression, sharp), the embedding lands in a different region
of the feature space compared to embeddings from low-resolution video
frames — even for the same person. Cosine similarity between a clean photo
and a blurry video crop of the same person is often 0.40-0.60, well below
what same-domain pairs achieve.

Fix: before extracting features, the registration photo is preprocessed to
simulate video-camera quality (downscale, blur, JPEG compression, lighting
jitter). Multiple augmented versions are generated and their embeddings are
averaged, producing a single robust descriptor that is closer to the video
domain. See also the lowered match_threshold in db_config.py.
"""

import logging
import random

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


def _simulate_video_quality(image: np.ndarray) -> np.ndarray:
    """Apply deterministic preprocessing to simulate low-quality video domain.

    The Re-ID model was trained on surveillance-style crops (~128x64 px).
    Professional photos are much larger, sharper, and have different color
    profiles. This function bridges the gap.
    """
    h, w = image.shape[:2]

    # 1. Downscale to typical detection-crop size (~128px height max).
    #    The model's deep branch resizes to 256x128 internally anyway, but
    #    the colour / texture / proportion cues operate on the raw pixels
    #    at whatever resolution they arrive — matching the scale helps.
    target_h = min(h, 160)
    if h > target_h:
        scale = target_h / h
        new_w = max(32, int(w * scale))
        image = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)

    # 2. Mild Gaussian blur — simulates camera defocus / motion blur
    image = cv2.GaussianBlur(image, (3, 3), 0.5)

    # 3. JPEG compression artifacts — video streams are compressed
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    success, enc = cv2.imencode(".jpg", image, encode_param)
    if success:
        image = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return image


def _random_augment(image: np.ndarray) -> np.ndarray:
    """Apply a random augmentation to simulate different viewing conditions."""
    aug = image.copy()

    # Random brightness / contrast jitter
    alpha = 1.0 + random.uniform(-0.25, 0.25)
    beta = random.randint(-35, 35)
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # Random blur (sometimes more, sometimes less)
    if random.random() > 0.5:
        k = random.choice([3, 5])
        blur_sigma = random.uniform(0.3, 1.0)
        aug = cv2.GaussianBlur(aug, (k, k), blur_sigma)

    # Random downscale + upscale (simulates variable resolution)
    if random.random() > 0.4:
        h, w = aug.shape[:2]
        factor = random.uniform(0.25, 0.65)
        small = cv2.resize(aug, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        aug = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return aug


def _load_image(image_path_or_array):
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
        if image is None:
            logger.warning(f"Could not read image: {image_path_or_array}")
        return image
    return image_path_or_array


def embed_image(image_path_or_array, num_augmentations: int = 5) -> np.ndarray:
    """
    Produce a robust 698-dim descriptor for a person image.

    To bridge the domain gap between high-resolution registration photos
    and low-resolution video frames, the image is:
      1. Preprocessed to simulate video quality (downscale, blur, JPEG
         compression).
      2. Augmented N times with random brightness/contrast/blur/resolution
         jitter, and an embedding is extracted from each version.
      3. The N embeddings are averaged and L2-normalised into one robust
         descriptor.

    This single averaged descriptor lives closer to the video-feature domain
    than any single high-res embedding would, improving matching success
    without requiring the user to use video screenshots for registration.

    Args:
        image_path_or_array: file path (str) or BGR numpy array.
        num_augmentations: how many augmented embeddings to average
            (default 5; 0 = use only the preprocessed base image, no
            random augmentation).

    Returns:
        698-dim float32 numpy array, or None if the image is unreadable.
    """
    image = _load_image(image_path_or_array)
    if image is None:
        return None

    h, w = image.shape[:2]
    min_size = EMBEDDING_SETTINGS["min_image_size"]
    if h < min_size or w < min_size:
        logger.warning(f"Image too small ({w}x{h}), skipping")
        return None

    engine = _get_engine()

    # Step 1: deterministic video-quality preprocessing
    preprocessed = _simulate_video_quality(image)

    # Step 2: collect embeddings — base + augmented versions
    embeddings = []

    # Always include the clean preprocessed version
    base_feat = engine.extract_feature(preprocessed, [0, 0, preprocessed.shape[1], preprocessed.shape[0]])
    if base_feat is not None:
        embeddings.append(base_feat)

    # Add randomly augmented versions
    for _ in range(num_augmentations):
        aug = _random_augment(preprocessed)
        feat = engine.extract_feature(aug, [0, 0, aug.shape[1], aug.shape[0]])
        if feat is not None:
            embeddings.append(feat)

    if not embeddings:
        logger.warning("Feature extraction failed for all augmented versions")
        return None

    # Step 3: average and re-normalise
    avg = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg)
    if norm < 1e-8:
        logger.warning("Averaged embedding is zero — something went wrong")
        return None
    avg /= norm

    logger.debug(
        f"Embedded with {len(embeddings)}/{1 + num_augmentations} "
        f"versions (base + {num_augmentations} augs)"
    )
    return avg


def embed_images(image_paths, num_augmentations: int = 5) -> list:
    """Embed a list of images, silently skipping any that fail."""
    embeddings = []
    for path in image_paths:
        feat = embed_image(path, num_augmentations=num_augmentations)
        if feat is not None:
            embeddings.append(feat)
    return embeddings
