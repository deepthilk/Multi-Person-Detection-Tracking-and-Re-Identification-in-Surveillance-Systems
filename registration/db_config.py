"""
Configuration for the Person Registration & Identity Database module.

This file is OWNED by the registration module and is intentionally kept
separate from the shared config.py so that editing registration settings
never touches a file Deepthi / Lekha / Pranjali also edit.
"""

# ==============================================================================
# STORAGE PATHS
# ==============================================================================

DB_SETTINGS = {
    # Where the identity database (name + embeddings + metadata) is persisted.
    "output_dir": "outputs/registration",
    "db_json": "outputs/registration/identity_db.json",

    # Where a copy of each registered image is stored, organised by person.
    "images_dir": "outputs/registration/images",
}

# ==============================================================================
# EMBEDDING SETTINGS
# ==============================================================================

EMBEDDING_SETTINGS = {
    # Reuses Deepthi's Re-ID backbone so embeddings stored here are directly
    # comparable (same 698-dim descriptor space) to the ones produced during
    # live Re-ID matching. See registration/embedder.py.
    "device": "cpu",          # 'cuda' or 'cpu' — registration is a one-off
                               # batch job, CPU is fine and avoids GPU
                               # contention with the live pipeline.
    "min_image_size": 20,     # reject images smaller than this on either side
}

# ==============================================================================
# SEARCH / MATCH SETTINGS
# ==============================================================================

SEARCH_SETTINGS = {
    # Cosine similarity threshold used by IdentityDatabase.match() when
    # comparing a query embedding (e.g. from a live Re-ID track) against
    # registered persons.
    "match_threshold": 0.55,
    "top_k": 3,
}
