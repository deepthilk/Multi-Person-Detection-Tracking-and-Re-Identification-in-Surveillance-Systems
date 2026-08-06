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

    def add_person(self, name: str, embeddings: list, image_paths: list = None,
                   face_embeddings: list = None):
        """
        Add or update a person.

        Args:
            name: unique display name, used as the lookup key.
            embeddings: list of 1-D numpy arrays / lists (one per image).
            image_paths: original source paths, stored as metadata only.
            face_embeddings: optional list of 512-dim face embeddings, one per
                image where a confident face was detected. Stored as the
                person's face gallery so search can match by face.
        """
        if not embeddings:
            raise ValueError(f"No usable embeddings for '{name}' — nothing to store")

        vectors = [np.asarray(e, dtype=np.float32).tolist() for e in embeddings]
        average = np.mean(np.array(vectors, dtype=np.float32), axis=0).tolist()

        face_embeddings = face_embeddings or []
        face_vectors = [np.asarray(f, dtype=np.float32).tolist() for f in face_embeddings]

        existing = self._data.get(name)
        if existing:
            # Registering more photos for someone already in the DB: append
            # rather than overwrite, and recompute the average.
            vectors = existing["embeddings"] + vectors
            average = np.mean(np.array(vectors, dtype=np.float32), axis=0).tolist()
            face_vectors = existing.get("face_embeddings", []) + face_vectors
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
            "metadata": {
                "registered_at": registered_at,
                "last_updated": datetime.now().isoformat(),
                "num_images": num_images,
                "image_paths": all_paths,
            },
        }
        self.save()
        logger.info(
            f"✅ Registered '{name}' with {num_images} total image(s) "
            f"({len(face_vectors)} face(s))"
        )

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

    def export_with_faces(self) -> dict:
        """
        Hand the whole database out with BOTH cues:
        {name: {"appearance": np.ndarray(698,), "faces": [np.ndarray(128,), ...]}}.

        `faces` is the per-person face gallery (may be empty for people
        registered before face support, or whose photos never showed a face).
        """
        return {
            name: {
                "appearance": np.asarray(record["average_embedding"], dtype=np.float32),
                "faces": [np.asarray(f, dtype=np.float32)
                          for f in (record.get("face_embeddings") or [])],
            }
            for name, record in self._data.items()
        }

    def match_multimodal(self, query_appearance, query_faces=None, top_k: int = None,
                         threshold: float = None) -> list:
        """
        Match a video identity against every registered person using BOTH
        body appearance and face (when the query side has faces).

        Score-level fusion:
          - No faces available on either side  -> appearance similarity only.
          - Face clearly confirms the person   -> face dominates (70/30), and
            the match threshold is the FACE threshold (face_confirmed_threshold),
            not the appearance one: once the face itself clearly says "same
            person" (measured separation: same-person >= 0.54, different-person
            <= 0.28), a weak body-appearance blend must not drag the result
            below the appearance threshold. Real case: v3 sid=2 matched deeps'
            face at 0.536 but app=0.483 -> blended 0.520, just under 0.55.
          - Face clearly disagrees             -> veto: a different face can't
            be overridden by coincidental clothing similarity.
          - Ambiguous face                     -> balanced 50/50 blend.

        Returns a list of dicts:
            {"name", "score", "appearance_sim", "face_sim", "cues"}
        sorted by score descending, filtered by threshold, capped at top_k.
        """
        top_k = top_k or SEARCH_SETTINGS["top_k"]
        threshold = threshold if threshold is not None else SEARCH_SETTINGS["match_threshold"]

        from reidentification.insight_face import InsightFaceExtractor

        def _fuse(app_sim: float, face_sim) -> tuple:
            s = SEARCH_SETTINGS
            if face_sim is None:
                return app_sim, ["appearance"], threshold
            if face_sim >= s["face_confirmed_threshold"]:
                score = (s["fused_weight_face_confirmed"] * face_sim +
                         s["fused_weight_appearance_confirmed"] * app_sim)
                return score, ["appearance", "face"], s["face_confirmed_threshold"]
            if face_sim < s["face_veto_threshold"]:
                return min(app_sim, face_sim), ["appearance", "face(veto)"], threshold
            return (s["fused_weight_face_ambiguous"] * face_sim +
                    s["fused_weight_appearance_ambiguous"] * app_sim), \
                   ["appearance", "face"], threshold

        query_faces = query_faces or []
        results = []
        for name, record in self._data.items():
            app_sim = _cosine(query_appearance, record["average_embedding"])
            gallery_faces = record.get("face_embeddings") or []
            face_sim = None
            if query_faces and gallery_faces:
                face_sim = max(
                    InsightFaceExtractor.similarity(q, g)
                    for q in query_faces for g in gallery_faces
                )
            score, cues, eff_thresh = _fuse(app_sim, face_sim)
            if score >= eff_thresh:
                results.append({
                    "name": name,
                    "score": round(float(score), 4),
                    "appearance_sim": round(float(app_sim), 4),
                    "face_sim": round(float(face_sim), 4) if face_sim is not None else None,
                    "cues": cues,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"IdentityDatabase({len(self._data)} person(s) @ {self.db_path})"
