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
import shutil
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

    def add_person(self, name: str, embeddings: list, image_paths: list = None, face_embeddings: list = None):
        """
        Add or update a person.

        Args:
            name: unique display name, used as the lookup key.
            embeddings: list of 1-D numpy arrays / lists (one per image).
            image_paths: original source paths, stored as metadata only.
            face_embeddings: optional list of face embedding vectors.
        """
        if not embeddings:
            raise ValueError(f"No usable embeddings for '{name}' — nothing to store")

        vectors = [np.asarray(e, dtype=np.float32).tolist() for e in embeddings]
        average = np.mean(np.array(vectors, dtype=np.float32), axis=0).tolist()

        face_vectors = [np.asarray(f, dtype=np.float32).tolist() for f in (face_embeddings or [])]
        avg_face = np.mean(np.array(face_vectors, dtype=np.float32), axis=0).tolist() if face_vectors else None

        existing = self._data.get(name)
        if existing:
            # Registering more photos for someone already in the DB: append
            # rather than overwrite, and recompute the average.
            vectors = existing["embeddings"] + vectors
            average = np.mean(np.array(vectors, dtype=np.float32), axis=0).tolist()

            existing_faces = existing.get("face_embeddings", [])
            face_vectors = existing_faces + face_vectors
            avg_face = np.mean(np.array(face_vectors, dtype=np.float32), axis=0).tolist() if face_vectors else existing.get("average_face_descriptor")

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
            "face_embeddings": face_vectors,
            "average_face_descriptor": avg_face,
            "metadata": {
                "registered_at": registered_at,
                "last_updated": datetime.now().isoformat(),
                "num_images": num_images,
                "image_paths": all_paths,
            },
        }
        self.save()
        logger.info(f"✅ Registered '{name}' with {num_images} total image(s) (faces: {len(face_vectors)})")

    def delete_person(self, name: str) -> bool:
        if name in self._data:
            del self._data[name]
            self.save()
            self._remove_person_images(name)
            return True
        return False

    def _remove_person_images(self, name: str):
        """Remove the on-disk photo folder for this person, if any.

        Deleting a person should not leave orphaned photos behind in
        ``outputs/registration/images/<name>/``.
        """
        images_dir = Path(DB_SETTINGS["images_dir"]) / name
        if images_dir.exists():
            try:
                shutil.rmtree(images_dir)
                logger.info("Removed photo folder for deleted person: %s", images_dir)
            except OSError as exc:
                logger.warning("Could not remove photo folder %s: %s", images_dir, exc)

    # ── reads ────────────────────────────────────────────────────────────

    def list_persons(self) -> list:
        return sorted(self._data.keys())

    def get_person(self, name: str) -> dict:
        return self._data.get(name)

    def person_exists(self, name: str) -> bool:
        return name.strip() in self._data

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

    def match(self, query_embedding, query_face_embedding=None, top_k: int = None, threshold: float = None) -> list:
        """
        Compare a query embedding against registered persons.

        Body-only (no face available): cosine against the average body
        embedding, threshold = match_threshold (0.55).

        Face available (query AND registered person both have a face):
          * face_sim >= face_match_threshold (0.40) -> the face CONFIRMS the
            identity on its own; the match score is the face similarity and
            the bar is the lower face threshold. This is what recognises a
            person who changed clothes / lighting between registration and
            the live clip — the body cue (which hates outfit changes) no
            longer drags a clear face below the body threshold.
          * face_sim < face_veto_threshold (0.30) -> confident mismatch;
            veto even a strong body score (two people can't share a face).
          * otherwise -> the face is inconclusive, so the BODY decides
            (threshold = match_threshold). An inconclusive face must not
            pull a strong body match below the bar, otherwise a different
            camera angle (smaller/partial faces) would suppress people who
            genuinely match by appearance.
        """
        top_k = top_k or SEARCH_SETTINGS["top_k"]
        body_threshold = threshold if threshold is not None else SEARCH_SETTINGS["match_threshold"]
        face_confirm = SEARCH_SETTINGS.get("face_match_threshold", 0.40)
        face_veto = SEARCH_SETTINGS.get("face_veto_threshold", 0.30)

        from reidentification.face_cue import FaceCueExtractor

        scores = []
        for name, record in self._data.items():
            body_sim = _cosine(query_embedding, record["average_embedding"])

            avg_face = record.get("average_face_descriptor")
            if query_face_embedding is not None and avg_face is not None:
                face_sim = FaceCueExtractor.similarity(avg_face, query_face_embedding)
                if face_sim is not None:
                    if face_sim < face_veto:
                        sim, bar = 0.0, face_veto
                    elif face_sim >= face_confirm:
                        sim, bar = face_sim, face_confirm
                    else:
                        sim, bar = body_sim, body_threshold
                else:
                    sim, bar = body_sim, body_threshold
            else:
                sim, bar = body_sim, body_threshold

            if sim >= bar:
                scores.append((name, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def cross_similarity(self, name: str) -> list:
        """
        Compare one registered person's average embeddings against every OTHER
        person in the database.

        Returns a list of dicts ``{"other", "body_sim", "face_sim"}`` sorted
        by body similarity (highest first). ``face_sim`` is ``None`` when
        either side lacks a face descriptor.

        This backs the registration-time guardrails: it answers "which
        existing registrant can the new person NOT be told apart from?" for
        the body cue, the face cue, or both.
        """
        record = self._data.get(name)
        if not record:
            return []
        avg_body = record.get("average_embedding")
        avg_face = record.get("average_face_descriptor")

        from reidentification.face_cue import FaceCueExtractor

        results = []
        for other, other_record in self._data.items():
            if other == name:
                continue
            body_sim = _cosine(avg_body, other_record["average_embedding"])
            face_sim = None
            other_face = other_record.get("average_face_descriptor")
            if avg_face is not None and other_face is not None:
                face_sim = FaceCueExtractor.similarity(avg_face, other_face)
            results.append({"other": other, "body_sim": body_sim, "face_sim": face_sim})
        results.sort(key=lambda x: x["body_sim"], reverse=True)
        return results

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

    # ── backup / restore ─────────────────────────────────────────────────

    def export_db(self, backup_path: str):
        """
        Write a standalone copy of the entire database (embeddings, images
        list, metadata — everything) to `backup_path`. Useful before a risky
        operation, or to hand a snapshot to a teammate for local testing.
        """
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with open(backup_path, "w") as f:
            json.dump(self._data, f, indent=2)
        logger.info(f"Backed up {len(self._data)} person(s) -> {backup_path}")

    def import_db(self, backup_path: str, merge: bool = True):
        """
        Restore from a backup produced by export_db().

        Args:
            merge: if True (default), backed-up persons are added on top of
                whatever is already in the live database (existing photos
                for the same name are combined via add_person's merge
                logic). If False, the backup completely replaces the
                current database.
        """
        with open(backup_path, "r") as f:
            backup_data = json.load(f)

        if not merge:
            self._data = backup_data
            self.save()
            logger.info(f"Restored {len(self._data)} person(s) from {backup_path} (replaced)")
            return

        for name, record in backup_data.items():
            embeddings = record["embeddings"]
            image_paths = record["metadata"].get("image_paths", [])
            self.add_person(name, embeddings, image_paths=image_paths)
        logger.info(f"Merged {len(backup_data)} person(s) from {backup_path}")

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"IdentityDatabase({len(self._data)} person(s) @ {self.db_path})"
