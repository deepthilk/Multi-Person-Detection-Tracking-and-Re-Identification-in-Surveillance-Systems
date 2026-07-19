"""
Unit tests for IdentityDatabase — these test the storage / search / match
logic in isolation using fake embeddings, so they run fast and don't need
the Re-ID model weights, torch, or GPU to be set up.

For a full end-to-end test (real images -> real embeddings), see
registration/validate_setup.py and register.py's CLI, which do exercise
the real embedder.

Run with:
    python -m registration.tests.test_identity_db
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from registration.identity_db import IdentityDatabase


def _fake_embedding(seed: int, dim: int = 698) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_add_and_list():
    tmp_dir = tempfile.mkdtemp()
    try:
        db = IdentityDatabase(db_path=str(Path(tmp_dir) / "identity_db.json"))
        db.add_person("Alice", [_fake_embedding(1), _fake_embedding(2)])
        db.add_person("Bob", [_fake_embedding(3)])

        assert db.list_persons() == ["Alice", "Bob"]
        assert db.get_person("Alice")["metadata"]["num_images"] == 2
        print("✅ test_add_and_list passed")
    finally:
        shutil.rmtree(tmp_dir)


def test_persistence_reload():
    tmp_dir = tempfile.mkdtemp()
    try:
        path = str(Path(tmp_dir) / "identity_db.json")
        db1 = IdentityDatabase(db_path=path)
        db1.add_person("Alice", [_fake_embedding(1)])

        db2 = IdentityDatabase(db_path=path)  # fresh load from disk
        assert db2.list_persons() == ["Alice"]
        print("✅ test_persistence_reload passed")
    finally:
        shutil.rmtree(tmp_dir)


def test_search_by_name():
    tmp_dir = tempfile.mkdtemp()
    try:
        db = IdentityDatabase(db_path=str(Path(tmp_dir) / "identity_db.json"))
        db.add_person("Alice", [_fake_embedding(1)])
        db.add_person("Alfred", [_fake_embedding(2)])
        db.add_person("Bob", [_fake_embedding(3)])

        results = [name for name, _ in db.search_by_name("al")]
        assert set(results) == {"Alice", "Alfred"}
        print("✅ test_search_by_name passed")
    finally:
        shutil.rmtree(tmp_dir)


def test_match_and_export_for_reid():
    tmp_dir = tempfile.mkdtemp()
    try:
        db = IdentityDatabase(db_path=str(Path(tmp_dir) / "identity_db.json"))
        alice_emb = _fake_embedding(1)
        db.add_person("Alice", [alice_emb])
        db.add_person("Bob", [_fake_embedding(2)])

        # Querying with something close to Alice's embedding should match Alice.
        query = alice_emb + np.random.default_rng(0).normal(scale=0.01, size=698).astype(np.float32)
        matches = db.match(query, threshold=0.0)
        assert matches[0][0] == "Alice"

        exported = db.export_for_reid()
        assert set(exported.keys()) == {"Alice", "Bob"}
        assert exported["Alice"].shape == (698,)
        print("✅ test_match_and_export_for_reid passed")
    finally:
        shutil.rmtree(tmp_dir)


def test_add_more_images_updates_average():
    tmp_dir = tempfile.mkdtemp()
    try:
        db = IdentityDatabase(db_path=str(Path(tmp_dir) / "identity_db.json"))
        db.add_person("Alice", [_fake_embedding(1)])
        db.add_person("Alice", [_fake_embedding(2)])  # more photos, same person

        record = db.get_person("Alice")
        assert record["metadata"]["num_images"] == 2
        assert len(record["embeddings"]) == 2
        print("✅ test_add_more_images_updates_average passed")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_add_and_list()
    test_persistence_reload()
    test_search_by_name()
    test_match_and_export_for_reid()
    test_add_more_images_updates_average()
    print("\nAll registration DB tests passed ✅")
