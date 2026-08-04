"""
High-level registration workflow.

This is the one function most people (including future-you writing the
CLI or Pranjali's upload form) need to call.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path

from registration.db_config import DB_SETTINGS, GUARDRAIL_SETTINGS
from registration.embedder import embed_images
from registration.identity_db import IdentityDatabase

logger = logging.getLogger(__name__)


def register_person(name: str, image_paths: list, db: IdentityDatabase = None,
                     overwrite: bool = False, num_augmentations: int = 5) -> dict:
    """
    Register a known person from one or more images.

    Steps:
      1. Generate a 698-dim embedding per image (via registration.embedder,
         which reuses Deepthi's Re-ID backbone so the vectors are directly
         comparable to live-camera embeddings).
      2. Copy the source images into outputs/registration/images/<name>/
         so the database can be rebuilt or audited later.
      3. Store name + embeddings + metadata in the identity database.

    Args:
        name: person's display name (also the lookup key — must be unique).
        image_paths: list of file paths to photos of this person.
        db: optional existing IdentityDatabase instance (mainly for tests /
            batch registration so the JSON file isn't reloaded every call).
        overwrite: if the name already exists and this is False (default),
            the new photos are ADDED to the existing person (their average
            embedding is recomputed over all photos, old + new). If True,
            the old record is deleted first, so only the new photos count.

    Returns:
        The stored record for this person (dict), or raises ValueError if
        none of the images produced a usable embedding.
    """
    if not name or not name.strip():
        raise ValueError("Person name cannot be empty")
    name = name.strip()

    if not image_paths:
        raise ValueError(f"No images provided for '{name}'")

    # NOTE: an empty IdentityDatabase is falsy (it defines __len__), so a
    # truthiness check here would silently drop a caller-supplied empty db
    # and fall back to the live database file. Always compare against None.
    db = db if db is not None else IdentityDatabase()

    if db.person_exists(name):
        if overwrite:
            _backup_before_overwrite(db)
            db.delete_person(name)
            logger.info(f"'{name}' already existed — replacing (--overwrite)")
        else:
            existing_count = db.get_person(name)["metadata"]["num_images"]
            logger.warning(
                f"'{name}' already has {existing_count} photo(s) registered. "
                f"Adding {len(image_paths)} more (pass overwrite=True to replace instead)."
            )

    logger.info(f"Registering '{name}' with {len(image_paths)} image(s) "
                f"(num_augmentations={num_augmentations})...")

    embeddings = embed_images(image_paths, num_augmentations=num_augmentations)
    if not embeddings:
        raise ValueError(
            f"None of the provided images for '{name}' produced a usable "
            f"embedding (check they exist, are readable, and are large enough)"
        )

    stored_paths = _copy_images(name, image_paths)

    face_embeddings = []
    try:
        from reidentification.face_cue import FaceCueExtractor
        import cv2
        face_extractor = FaceCueExtractor()
        for path in stored_paths:
            img = cv2.imread(path)
            if img is not None:
                face_feat = face_extractor.extract(img, [0, 0, img.shape[1], img.shape[0]])
                if face_feat is not None:
                    face_embeddings.append(face_feat)
    except Exception as e:
        logger.warning(f"Could not extract face descriptors during registration: {e}")

    if not face_embeddings:
        logger.warning(
            f"No face descriptor was extracted from any photo of '{name}'. "
            f"This registration is body-only. For uniformed subjects the body "
            f"cue is weak (different people measured 0.90+ body similarity), so "
            f"re-identifying '{name}' will rely purely on appearance and may "
            f"confuse them with other registrants. Add at least one clear, "
            f"frontal face photo if possible."
        )
    elif len(face_embeddings) < len(stored_paths):
        logger.warning(
            f"Only {len(face_embeddings)}/{len(stored_paths)} photos of '{name}' "
            f"yielded a face descriptor — the rest are body-only samples."
        )

    db.add_person(name, embeddings, image_paths=stored_paths, face_embeddings=face_embeddings)
    _warn_discrimination(db, name)

    logger.info(
        f"✅ '{name}' registered: {len(embeddings)}/{len(image_paths)} images used"
    )
    return db.get_person(name)


def _backup_before_overwrite(db: IdentityDatabase):
    """Take a timestamped full-DB backup before a destructive overwrite, so a
    mis-click can always be rolled back with IdentityDatabase.import_db()."""
    if len(db) == 0:
        return
    backups_dir = Path(GUARDRAIL_SETTINGS["backups_dir"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backups_dir / f"identity_db_{stamp}.json"
    db.export_db(str(backup_path))
    logger.warning(f"Destructive overwrite — previous DB backed up to {backup_path}")


def _warn_discrimination(db: IdentityDatabase, name: str):
    """After (re)registering, flag registrants the cues cannot distinguish from
    the new person. Warnings only — the registration still succeeds, but the
    operator now knows which identities will confuse the matcher at run time
    (and the self-check report will repeat these numbers)."""
    body_bar = GUARDRAIL_SETTINGS["body_discrimination_threshold"]
    face_bar = GUARDRAIL_SETTINGS["face_discrimination_threshold"]
    for row in db.cross_similarity(name):
        body_flag = row["body_sim"] >= body_bar
        face_flag = row["face_sim"] is not None and row["face_sim"] >= face_bar
        if not body_flag and not face_flag:
            continue
        if face_flag:
            logger.warning(
                f"'{name}' and '{row['other']}' share a near-identical face "
                f"descriptor (face similarity {row['face_sim']:.2f} >= "
                f"{face_bar}) — the same person registered twice under two "
                f"names, or a duplicated photo set."
            )
        elif row["face_sim"] is None:
            logger.warning(
                f"'{name}' (body-only) and '{row['other']}' cannot be separated "
                f"by appearance (body similarity {row['body_sim']:.2f} >= "
                f"{body_bar}) — faces are required for reliable matching."
            )
        else:
            logger.warning(
                f"Body cue alone cannot tell '{name}' from '{row['other']}' "
                f"(body similarity {row['body_sim']:.2f} >= {body_bar}, face "
                f"similarity {row['face_sim']:.2f} < {face_bar}) — faces will "
                f"be the only discriminating signal."
            )


def _copy_images(name: str, image_paths: list) -> list:
    """Copy source images into outputs/registration/images/<name>/ and
    return the new paths, so the registered dataset is self-contained and
    survives even if the original upload location is cleaned up later."""
    dest_dir = Path(DB_SETTINGS["images_dir"]) / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    stored = []
    for i, src in enumerate(image_paths):
        src_path = Path(src)
        if not src_path.exists():
            continue
        dest = dest_dir / f"{i:03d}_{src_path.name}"
        try:
            shutil.copy2(src_path, dest)
            stored.append(str(dest))
        except Exception as e:
            logger.warning(f"Could not copy {src_path} -> {dest}: {e}")
    return stored
