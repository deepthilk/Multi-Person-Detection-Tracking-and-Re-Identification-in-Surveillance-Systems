"""
Unit tests for the registration guardrails added in the hardening pass:

  * deterministic embeddings (same image -> identical embedding every run)
  * automatic backup before a destructive overwrite
  * face-presence warning (body-only registration)
  * cross-similarity discrimination warning
  * self-check report structure

These tests are designed to run WITHOUT torch / the Re-ID weights: the real
embedder is swapped for a tiny fake engine, and face extraction is stubbed.
The only heavy dependency they need is OpenCV (used by the embedder for the
video-quality preprocessing, which is pure cv2/numpy).

Run with:
    python -m registration.tests.test_guardrails
"""

import io
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from registration import embedder
from registration import register_person as rp
from registration.db_config import GUARDRAIL_SETTINGS
from registration.identity_db import IdentityDatabase


def _fake_embedding(seed: int, dim: int = 698) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _synthetic_image(size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return img


class _FakeEngine:
    """Deterministic stand-in for ReIDEngine.extract_feature(image, bbox)."""

    def extract_feature(self, image, bbox):
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        v = np.resize(crop.astype(np.float32), 698)
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else None


class _NoFaces:
    """FaceCueExtractor stand-in that never finds a face."""

    def __init__(self, *args, **kwargs):
        pass

    def extract(self, frame, bbox):
        return None


class _LogCapture:
    """Collect log records for the module under test."""

    def __init__(self, logger):
        self.records = []
        self._handler = logging.Handler()
        self._handler.emit = lambda record: self.records.append(record)
        self._logger = logger

    def __enter__(self):
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        return False

    def has_message_containing(self, text: str) -> bool:
        return any(text in r.getMessage() for r in self.records)


# ── deterministic embedding ────────────────────────────────────────────────

def test_embedding_deterministic_and_seed_sensitive():
    original_engine = embedder._engine
    try:
        embedder._engine = _FakeEngine()
        img = _synthetic_image()

        a = embedder.embed_image(img, num_augmentations=5, seed=42)
        b = embedder.embed_image(img, num_augmentations=5, seed=42)
        c = embedder.embed_image(img, num_augmentations=5, seed=123)

        assert a is not None and b is not None and c is not None
        assert np.array_equal(a, b), "same image + same seed must give identical embedding"
        assert not np.array_equal(a, c), "different seed must change the augmentation path"
        print(" test_embedding_deterministic_and_seed_sensitive passed")
    finally:
        embedder._engine = original_engine


# ── backup before overwrite ────────────────────────────────────────────────

def test_backup_before_overwrite():
    tmp = Path(tempfile.mkdtemp())
    try:
        backups = tmp / "backups"
        images = tmp / "images"
        images.mkdir()

        old_backups = GUARDRAIL_SETTINGS["backups_dir"]
        old_images = rp.DB_SETTINGS["images_dir"]
        GUARDRAIL_SETTINGS["backups_dir"] = str(backups)
        rp.DB_SETTINGS["images_dir"] = str(images)
        try:
            photo = tmp / "alice.jpg"
            photo.write_bytes(b"not a real jpeg, copy only")

            db = IdentityDatabase(db_path=str(tmp / "identity_db.json"))
            orig_embed = rp.embed_images
            rp.embed_images = lambda paths, num_augmentations=5: [_fake_embedding(1)]
            try:
                rp.register_person("Alice", [str(photo)], db=db)
                assert db.person_exists("Alice")
                assert not list(backups.glob("*.json")), "no backup on a fresh add"

                rp.register_person("Alice", [str(photo)], db=db, overwrite=True)
                bak = list(backups.glob("identity_db_*.json"))
                assert bak, "overwrite=True must auto-backup the previous DB"
                assert db.person_exists("Alice")
                print(" test_backup_before_overwrite passed")
            finally:
                rp.embed_images = orig_embed
        finally:
            GUARDRAIL_SETTINGS["backups_dir"] = old_backups
            rp.DB_SETTINGS["images_dir"] = old_images
    finally:
        shutil.rmtree(tmp)


# ── face-presence warning ───────────────────────────────────────────────────

def test_warns_when_no_face_found():
    tmp = Path(tempfile.mkdtemp())
    try:
        images = tmp / "images"
        images.mkdir()
        old_images = rp.DB_SETTINGS["images_dir"]
        rp.DB_SETTINGS["images_dir"] = str(images)
        old_face_class = None
        try:
            import reidentification.face_cue as fc
            old_face_class = fc.FaceCueExtractor
            fc.FaceCueExtractor = _NoFaces

            photo = tmp / "person.jpg"
            photo.write_bytes(b"bytes")

            db = IdentityDatabase(db_path=str(tmp / "identity_db.json"))
            orig_embed = rp.embed_images
            rp.embed_images = lambda paths, num_augmentations=5: [_fake_embedding(7)]
            try:
                with _LogCapture(rp.logger) as cap:
                    rp.register_person("P1", [str(photo)], db=db)
                assert cap.has_message_containing("No face descriptor"), \
                    "body-only registration must raise a face-presence warning"
                print(" test_warns_when_no_face_found passed")
            finally:
                rp.embed_images = orig_embed
        finally:
            rp.DB_SETTINGS["images_dir"] = old_images
            if old_face_class is not None:
                import reidentification.face_cue as fc
                fc.FaceCueExtractor = old_face_class
    finally:
        shutil.rmtree(tmp)


# ── cross-similarity discrimination warning ────────────────────────────────

def test_warns_on_confusable_persons():
    tmp = Path(tempfile.mkdtemp())
    try:
        db = IdentityDatabase(db_path=str(tmp / "identity_db.json"))
        near_identical = _fake_embedding(1)
        db.add_person("A", [near_identical])
        db.add_person("B", [near_identical.copy()])  # body cue cannot separate
        db.add_person("C", [_fake_embedding(2)])     # clearly different body

        with _LogCapture(rp.logger) as cap:
            rp._warn_discrimination(db, "B")

        flagged = [r for r in cap.records if "B" in r.getMessage() and "A" in r.getMessage()]
        assert flagged, "B vs A (near-identical bodies) must be flagged"
        for r in flagged:
            assert "cannot be separated" in r.getMessage() or "Body cue" in r.getMessage()
        for r in cap.records:
            assert "C" not in r.getMessage(), "C is clearly different — must NOT be flagged"
        print(" test_warns_on_confusable_persons passed")
    finally:
        shutil.rmtree(tmp)


# ── self-check report structure ────────────────────────────────────────────

def test_self_check_report_structure():
    tmp = Path(tempfile.mkdtemp())
    try:
        db = IdentityDatabase(db_path=str(tmp / "identity_db.json"))
        db.add_person("Alice", [_fake_embedding(1), _fake_embedding(2)])
        db.add_person("Bob", [_fake_embedding(3)])

        from registration.self_check import run_self_check

        report = run_self_check(db=db, out_path=str(tmp / "report.json"))
        assert report["n_persons"] == 2
        assert len(report["per_person"]) == 2
        assert set(report["body_cross_sim_matrix"].keys()) == {"Alice", "Bob"}
        assert set(report["face_cross_sim_matrix"].keys()) == {"Alice", "Bob"}
        assert "self_consistency" in report
        assert "summary" in report["self_consistency"]
        assert report["self_consistency"]["summary"]["total_photos"] == 0, \
            "no stored photos -> nothing re-embedded (model not required)"
        assert "guardrail_flags" in report and "caveat" in report
        assert (tmp / "report.json").exists()

        # deterministic report: identical matrices across runs
        report2 = run_self_check(db=db)
        assert report["body_cross_sim_matrix"] == report2["body_cross_sim_matrix"]
        print(" test_self_check_report_structure passed")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_embedding_deterministic_and_seed_sensitive()
    test_backup_before_overwrite()
    test_warns_when_no_face_found()
    test_warns_on_confusable_persons()
    test_self_check_report_structure()
    print("\nAll registration guardrail tests passed")
