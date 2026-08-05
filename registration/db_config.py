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

    # Face + appearance score-level fusion (IdentityDatabase.match_multimodal).
    # Face similarity uses FaceCueExtractor.similarity (0..1, higher = same
    # person; face_recognition's conventional same-person Euclidean cutoff
    # ~0.6 maps to ~0.33, so "confirmed" is set comfortably above that).
    "face_confirmed_threshold": 0.40,   # >= this: faces clearly say "same person"
    "face_veto_threshold": 0.20,        # <  this: faces clearly say "different person"
    "fused_weight_face_confirmed": 0.70,    # face dominates a confirmed match
    "fused_weight_appearance_confirmed": 0.30,
    "fused_weight_face_ambiguous": 0.50,    # balanced when the face is ambiguous
    "fused_weight_appearance_ambiguous": 0.50,
}
