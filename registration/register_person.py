"""
High-level registration workflow.

This is the one function most people (including future-you writing the
CLI or Pranjali's upload form) need to call.
"""

import logging
import shutil
from pathlib import Path

from registration.db_config import DB_SETTINGS
from registration.embedder import embed_images
from registration.identity_db import IdentityDatabase

logger = logging.getLogger(__name__)


def register_person(name: str, image_paths: list, db: IdentityDatabase = None) -> dict:
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

    Returns:
        The stored record for this person (dict), or raises ValueError if
        none of the images produced a usable embedding.
    """
    if not name or not name.strip():
        raise ValueError("Person name cannot be empty")
    name = name.strip()

    if not image_paths:
        raise ValueError(f"No images provided for '{name}'")

    logger.info(f"Registering '{name}' with {len(image_paths)} image(s)...")

    embeddings = embed_images(image_paths)
    if not embeddings:
        raise ValueError(
            f"None of the provided images for '{name}' produced a usable "
            f"embedding (check they exist, are readable, and are large enough)"
        )

    stored_paths = _copy_images(name, image_paths)

    db = db or IdentityDatabase()
    db.add_person(name, embeddings, image_paths=stored_paths)

    logger.info(
        f"✅ '{name}' registered: {len(embeddings)}/{len(image_paths)} images used"
    )
    return db.get_person(name)


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
