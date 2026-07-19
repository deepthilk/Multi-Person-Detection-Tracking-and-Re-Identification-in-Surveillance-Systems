"""
Identity Database
=================

Stores, for every registered known person:
  - their name (unique key)
  - one embedding per uploaded image (698-dim, same space as live Re-ID)
  - an average embedding (used for fast matching)
  - metadata (when registered, how many images, source image paths)

Persisted as plain JSON so it's easy to inspect, back up, and diff in git
review — no binary formats, no database server required for a student
project of this size.

This file owns `outputs/registration/identity_db.json` exclusively.
No other module reads or writes it directly; everyone else goes through
the functions in this file.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from registration.db_config import DB_SETTINGS, SEARCH_SETTINGS

logger = logging.getLogger(__name__)


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


class IdentityDatabase:
    """Loads the JSON store on init; call .save() after any write."""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or DB_SETTINGS["db_json"])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = {}
        self.load()

    # ── persistence ──────────────────────────────────────────────────────

    def load(self):
        if self.db_path.exists():
            with open(self.db_path, "r") as f:
                self._data = json.load(f)
            logger.info(f"Loaded identity DB: {len(self._data)} person(s) from {self.db_path}")
        else:
            self._data = {}

    def save(self):
        with open(self.db_path, "w") as f:
            json.dump(self._data, f, indent=2)
        logger.info(f"Saved identity DB: {len(self._data)} person(s) -> {self.db_path}")

    # ── writes ───────────────────────────────────────────────────────────

    def add_person(self, name: str, embeddings: list, image_paths: list = None):
        """
        Add or update a person.

        Args:
            name: unique display name, used as the lookup key.
            embeddings: list of 1-D numpy arrays / lists (one per image).
            image_paths: original source paths, stored as metadata only.
        """
        if not embeddings:
            raise ValueError(f"No usable embeddings for '{name}' — nothing to store")

        vectors = [np.asarray(e, dtype=np.float32).tolist() for e in embeddings]
        average = np.mean(np.array(vectors, dtype=np.float32), axis=0).tolist()

        existing = self._data.get(name)
        if existing:
            # Registering more photos for someone already in the DB: append
            # rather than overwrite, and recompute the average.
            vectors = existing["embeddings"] + vectors
            average = np.mean(np.array(vectors, dtype=np.float32), axis=0).tolist()
            num_images = existing["metadata"]["num_images"] + len(embeddings)
            all_paths = existing["metadata"].get("image_paths", []) + (image_paths or [])
            registered_at = existing["metadata"]["registered_at"]
        else:
            num_images = len(embeddings)
            all_paths = image_paths or []
            registered_at = datetime.now().isoformat()

        self._data[name] = {
            "embeddings": vectors,
            "average_embedding": average,
            "metadata": {
                "registered_at": registered_at,
                "last_updated": datetime.now().isoformat(),
                "num_images": num_images,
                "image_paths": all_paths,
            },
        }
        self.save()
        logger.info(f"✅ Registered '{name}' with {num_images} total image(s)")

    def delete_person(self, name: str) -> bool:
        if name in self._data:
            del self._data[name]
            self.save()
            return True
        return False

    # ── reads ────────────────────────────────────────────────────────────

    def list_persons(self) -> list:
        return sorted(self._data.keys())

    def get_person(self, name: str) -> dict:
        return self._data.get(name)

    def search_by_name(self, query: str) -> list:
        """
        Case-insensitive substring search. Returns a list of
        (name, record) pairs so a partial query like "al" can match
        "Alice" and "Alfred".
        """
        q = query.strip().lower()
        return [
            (name, record) for name, record in self._data.items()
            if q in name.lower()
        ]

    def match(self, query_embedding, top_k: int = None, threshold: float = None) -> list:
        """
        Compare a query embedding (e.g. from a live Re-ID track) against
        every registered person's average embedding.

        Returns a list of (name, similarity) sorted by similarity
        descending, filtered by threshold, capped at top_k.

        This is the function Deepthi's Re-ID module (or Pranjali's
        dashboard) calls in Phase 3 to answer "who is this?".
        """
        top_k = top_k or SEARCH_SETTINGS["top_k"]
        threshold = threshold if threshold is not None else SEARCH_SETTINGS["match_threshold"]

        scores = []
        for name, record in self._data.items():
            sim = _cosine(query_embedding, record["average_embedding"])
            if sim >= threshold:
                scores.append((name, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def export_for_reid(self) -> dict:
        """
        Hand the whole database to the Re-ID module in a simple,
        ready-to-use form: {name: np.ndarray(698,)}.

        This is the single hand-off point described in the team plan
        ("Provide the database to the Re-ID module"). Deepthi's
        integration code only needs to call this — it never reads
        identity_db.json directly.
        """
        return {
            name: np.asarray(record["average_embedding"], dtype=np.float32)
            for name, record in self._data.items()
        }

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"IdentityDatabase({len(self._data)} person(s) @ {self.db_path})"
