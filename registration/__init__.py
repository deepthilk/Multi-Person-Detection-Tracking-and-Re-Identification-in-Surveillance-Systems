"""
Person Registration & Identity Database module.

Owned entirely by Prajna. Nobody else edits files inside this folder.

Public API (this is the contract other teammates / Phase-3 integration
code should rely on — the internals can change freely as long as these
keep working):

    from registration.register_person import register_person
    from registration.identity_db import IdentityDatabase

    # Register a known person from a set of images
    register_person("Alice", ["photos/alice_1.jpg", "photos/alice_2.jpg"])

    # Look someone up
    db = IdentityDatabase()
    record = db.search_by_name("Alice")

    # Hand the whole database to the Re-ID module (Phase 3)
    known_embeddings = db.export_for_reid()   # {"Alice": np.ndarray(698,), ...}
"""
