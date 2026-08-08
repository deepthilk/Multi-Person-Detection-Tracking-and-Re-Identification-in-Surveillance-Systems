"""
Phase 3 hand-off acceptance test.

Unlike test_identity_db.py (which only tests this module in isolation),
this test imports Deepthi's ACTUAL merged `reidentification.cross_camera_match`
and exercises the real hand-off: does `IdentityDatabase().export_for_reid()`
produce something `resolve_names()` can consume correctly, end-to-end?

This needs the repo's other modules to be importable (numpy + scipy is
enough — no GPU/torch/model download required, since we only touch the
name-matching logic, not feature extraction).

Run with:
    python -m registration.tests.test_phase3_handoff
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from registration.identity_db import IdentityDatabase


def test_export_for_reid_matches_cross_camera_match_contract():
    """
    Simulates what run_integrated_pipeline.py actually does:
      1. Load the identity database, export_for_reid().
      2. Feed it into Deepthi's real resolve_names(), with a fake
         "global descriptor" that's a close match to a registered person.
      3. Confirm the right name comes back.
    """
    from reidentification.cross_camera_match import resolve_names

    tmp_dir = tempfile.mkdtemp()
    try:
        db = IdentityDatabase(db_path=str(Path(tmp_dir) / "identity_db.json"))

        rng = np.random.default_rng(42)
        alice_embedding = rng.normal(size=698).astype(np.float32)
        alice_embedding /= np.linalg.norm(alice_embedding)
        bob_embedding = rng.normal(size=698).astype(np.float32)
        bob_embedding /= np.linalg.norm(bob_embedding)

        db.add_person("Alice", [alice_embedding])
        db.add_person("Bob", [bob_embedding])

        registered_persons = db.export_for_reid()
        assert set(registered_persons.keys()) == {"Alice", "Bob"}
        assert registered_persons["Alice"]["average_embedding"].shape == (698,)

        # A "global descriptor" from cross-camera matching that's a close
        # (but not identical) match to Alice — simulates a real camera
        # view of her from a different angle/lighting.
        noisy_alice = alice_embedding + rng.normal(scale=0.02, size=698).astype(np.float32)

        global_descriptors = {1: noisy_alice, 2: bob_embedding}
        name_map = resolve_names(global_descriptors, registered_persons)

        assert name_map[1]["name"] == "Alice"
        assert name_map[2]["name"] == "Bob"
        print("✅ test_export_for_reid_matches_cross_camera_match_contract passed")
        print(f"   Alice matched with similarity {name_map[1]['similarity']}")
        print(f"   Bob matched with similarity {name_map[2]['similarity']}")
    finally:
        shutil.rmtree(tmp_dir)


def test_unregistered_person_stays_unnamed():
    """A global identity with no close match in the DB should NOT be
    force-matched to the nearest (but wrong) registered person."""
    from reidentification.cross_camera_match import resolve_names

    tmp_dir = tempfile.mkdtemp()
    try:
        db = IdentityDatabase(db_path=str(Path(tmp_dir) / "identity_db.json"))
        rng = np.random.default_rng(1)
        alice_embedding = rng.normal(size=698).astype(np.float32)
        alice_embedding /= np.linalg.norm(alice_embedding)
        db.add_person("Alice", [alice_embedding])

        registered_persons = db.export_for_reid()

        # A totally unrelated random descriptor — a stranger, not Alice.
        stranger = rng.normal(size=698).astype(np.float32)
        stranger /= np.linalg.norm(stranger)

        name_map = resolve_names({1: stranger}, registered_persons)
        assert 1 not in name_map, "Stranger should not be matched to Alice"
        print("✅ test_unregistered_person_stays_unnamed passed")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_export_for_reid_matches_cross_camera_match_contract()
    test_unregistered_person_stays_unnamed()
    print("\nAll Phase 3 hand-off tests passed ✅")
