"""
Face-based identity database.
================================

WHY THIS EXISTS
---------------
The existing registration DB (identity_db.json) stores 698-dim BODY
descriptors produced by the fine-tuned ResNet-50 Re-ID model. That model
is trained on only ~16 identities, so its embeddings are NOT reliably
discriminative - different people can score higher than the same person
in different frames. This is what causes wrong matches / false positives
when searching.

Faces don't have that problem. face_recognition's 128-dim embeddings are
trained on millions of faces and have a well-calibrated Euclidean-distance
cutoff (~0.6): same person is almost always below it, different people are
almost always above it. Matching on faces gives near-zero false positives
on footage where the face is visible (which this project's demo footage is).

This module stores one person's face encodings alongside their registered
photos, so a photo-based search can match by FACE first (authoritative)
instead of by the unreliable body descriptor.

USAGE
-----
    from registration.face_db import FaceIdentityDB, build_from_registration

    db = FaceIdentityDB()                     # loads outputs/registration/face_db.json
    added = build_from_registration(db)       # (re)build from outputs/registration/images/<name>/
    db.get_person("Alice")                    # -> {face_encodings: [...], image_paths: [...]}
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from registration.db_config import DB_SETTINGS
from reidentification.face_encoder import (
    get_face_encoder, face_distance, distance_to_similarity,
)

logger = logging.getLogger(__name__)

FACE_DB_PATH = Path("outputs/registration/face_db.json")

# Euclidean distance between two 128-d face encodings. face_recognition's own
# default cutoff is 0.6; 0.55 is deliberately stricter so that a match means
# "confidently the same person" - almost zero false positives.
FACE_MATCH_DISTANCE = 0.55

# Stricter, footage-calibrated cutoff. Measured on this project's own tracking
# crops: same-person (intra-track) face distances median ~0.31, different
# people (inter-track) median ~0.54. 0.55 therefore sweeps in most people as
# false positives, so the effective search threshold is tightened to 0.40 and
# every reported segment must also contain >= MIN_SEGMENT_FRAMES confirmed
# frames. Accepts false negatives (face too small/turned away) over false
# positives, which is the safe direction for a surveillance tool.
FACE_CONFIRM_DISTANCE = 0.40

# Minimum number of matched frames a segment needs to be reported. A single
# frame near threshold is almost always noise; a real appearance yields a
# continuous run of matching frames.
MIN_SEGMENT_FRAMES = 3

# Min side length (px) of a detected face box we are willing to embed. Smaller
# faces are usually blurry and produce noisy encodings.
MIN_FACE_PIXELS = 20

# Multi-anchor gallery: every registered photo is ALSO embedded under a few
# mild augmentations (horizontal flip, brightness up/down, +/-8 degree rotation).
# A SINGLE registration photo therefore becomes 6-10 anchors that cover pose /
# lighting / colour variance, which is what lets one photo taken anywhere match
# footage shot under different conditions (cross-context / any-video tracking).
FACE_AUGMENT_ENABLED = True

# An augmented anchor is only kept when it stays this close (encoder distance)
# to its source photo. Guards against garbage anchors (e.g. a rotation that
# pushed the face into a black corner or a blurry variant).
AUGMENT_ANCHOR_MAX_DIST = 0.35


def _augment_views(image):
    """Yield (label, augmented_image) mild variants of a registration photo."""
    views = []
    h, w = image.shape[:2]
    views.append(("hflip", cv2.flip(image, 1)))
    for label, alpha in (("bright+", 1.25), ("bright-", 0.80)):
        views.append((label, cv2.convertScaleAbs(image, alpha=alpha, beta=0)))
    center = (w / 2.0, h / 2.0)
    for label, angle in (("rot+", 8.0), ("rot-", -8.0)):
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        views.append((label, cv2.warpAffine(
            image, m, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE)))
    return views


def extract_face_encodings(image, max_faces=1, min_face_px=MIN_FACE_PIXELS, upsample=1):
    """
    Detect faces in an image and return their encodings using the ACTIVE
    face encoder (ArcFace 512-dim, or dlib 128-dim).

    Args:
        image: BGR image (numpy array).
        max_faces: how many faces to return (largest first).
        min_face_px: skip faces smaller than this on their shorter side.
        upsample: kept for API compatibility (dlib-only knob).

    Returns:
        list of np.ndarray encodings, best faces first.
        Empty list if no face is confidently found.
    """
    if image is None:
        return []
    try:
        return get_face_encoder().encode_image_faces(
            image, max_faces=max_faces, min_face_px=min_face_px)
    except Exception as e:
        logger.warning(f"Face extraction error: {e}")
        return []


# NOTE: `face_distance` and `distance_to_similarity` are imported from
# reidentification.face_encoder above, so every consumer automatically uses
# the active encoder's distance metric (Euclidean for dlib, cosine for
# ArcFace).


class FaceIdentityDB:
    """Persistent store of per-person face encodings (JSON at face_db.json)."""

    def __init__(self, path=FACE_DB_PATH):
        self.path = Path(path)
        self._data = self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read {self.path}: {e}")
                return {}
            self._check_encoder_compat(data)
            return data
        return {}

    def _check_encoder_compat(self, data):
        """Warn + neutralize encodings stored with a different face encoder.

        Mixing embedding spaces (e.g. 128-d dlib Euclidean vs 512-d ArcFace
        cosine) makes distance comparisons meaningless, so incompatible
        encodings are dropped in-memory (the JSON is untouched) until the DB
        is rebuilt with the active encoder."""
        active = get_face_encoder().name
        for name, rec in data.items():
            if not isinstance(rec, dict):
                continue
            stored = rec.get("encoder", "dlib")
            if stored != active:
                n = rec.get("num_faces", len(rec.get("face_encodings", [])))
                logger.error(
                    f"Face DB entry '{name}' was built with '{stored}' but the "
                    f"active encoder is '{active}' - ignoring its {n} encoding(s). "
                    f"Rebuild with:  python register.py face-db --rebuild")
                rec["face_encodings"] = []
                rec["num_faces"] = 0

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def add_person(self, name, encodings, image_paths):
        """Store a person's face encodings (list of np arrays or lists)."""
        self._data[name] = {
            "face_encodings": [e.tolist() if isinstance(e, np.ndarray) else list(e)
                               for e in encodings],
            "num_faces": len(encodings),
            "image_paths": [str(p) for p in image_paths],
            "encoder": get_face_encoder().name,
        }
        self.save()
        return self._data[name]

    def get_person(self, name):
        rec = self._data.get(name)
        if rec is None:
            return None
        return {
            "face_encodings": [np.asarray(e, dtype=np.float32) for e in rec["face_encodings"]],
            "num_faces": rec.get("num_faces", len(rec["face_encodings"])),
            "image_paths": rec.get("image_paths", []),
        }

    def list_persons(self):
        return list(self._data.keys())

    def encodings_for(self, name):
        rec = self.get_person(name)
        return rec["face_encodings"] if rec else []

    def __len__(self):
        return len(self._data)


def build_from_registration(face_db=None, image_exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    (Re)build the face DB by scanning every registered person's copied images
    under outputs/registration/images/<name>/, embedding them with the ACTIVE
    face encoder (ArcFace 512-dim by default). One encoding per image (the
    largest face found). Returns [(name, num_faces), ...] for people whose
    faces could be extracted.
    """
    encoder = get_face_encoder()
    face_db = face_db or FaceIdentityDB()
    # True rebuild: the images dir is the single source of truth, so start
    # fresh each time. Otherwise deleted/re-registered persons linger forever.
    face_db._data = {}
    images_dir = Path(DB_SETTINGS["images_dir"])
    if not images_dir.exists():
        logger.warning(f"Registration images dir not found: {images_dir}")
        return []

    added = []
    for person_dir in sorted(images_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        image_paths = [p for p in sorted(person_dir.iterdir())
                       if p.suffix.lower() in image_exts]
        if not image_paths:
            logger.warning(f"'{name}': no images found, skipping")
            continue
        encodings = []
        used_paths = []
        aug_added = 0
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            face_encs = extract_face_encodings(img, max_faces=1)
            if face_encs:
                encodings.append(face_encs[0])
                used_paths.append(str(img_path))
                # Multi-anchor gallery: keep mild augmented variants that stay
                # close to the source photo (see AUGMENT_ANCHOR_MAX_DIST).
                if FACE_AUGMENT_ENABLED:
                    src = face_encs[0]
                    for label, aug in _augment_views(img):
                        aug_encs = extract_face_encodings(aug, max_faces=1)
                        if not aug_encs:
                            continue
                        d = encoder.distance(src, aug_encs[0])
                        if d <= AUGMENT_ANCHOR_MAX_DIST:
                            encodings.append(aug_encs[0])
                            aug_added += 1
        if encodings:
            face_db.add_person(name, encodings, used_paths)
            added.append((name, len(encodings)))
            logger.info(f"Face DB: '{name}' - {len(encodings)} anchor(s) "
                        f"({len(image_paths)} image(s), {aug_added} augmented)")
        else:
            logger.warning(f"'{name}': NO face found in any of {len(image_paths)} image(s) - "
                           f"cannot search this person reliably")
    return added
