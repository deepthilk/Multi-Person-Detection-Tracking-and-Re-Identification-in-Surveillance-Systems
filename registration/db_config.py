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
    # Face similarity uses InsightFaceExtractor.similarity (0..1 cosine on
    # ArcFace w600k_mbf embeddings; measured on this project's footage:
    # same-person >= ~0.54, different-person <= ~0.28). 0.42 keeps a wide
    # margin above the different-person ceiling (0.28/0.30) while still
    # catching true same-person matches whose face quality is mediocre
    # (Prajna measured 0.44-0.47 on this footage — below the old 0.45 bar).
    "face_confirmed_threshold": 0.42,   # >= this: faces clearly say "same person"
    # A single lucky face-pair must not confirm a name. face_sim is the MAX
    # over all (query x gallery) pairs, so one stray pair can cross the
    # confirmed bar while the track's typical face is far below it (real case:
    # an unregistered Lekha scored max=0.579 vs deeps but median per-face best
    # was only 0.109, 2/30 faces >= 0.42). Require the MEDIAN of each query
    # face's best gallery similarity to also agree at the same-person level.
    # Measured on this footage: true matches median 0.55-0.70, this false
    # positive 0.109 — 0.30 sits safely between.
    "face_robust_median_threshold": 0.30,
    "face_veto_threshold": 0.30,        # <  this: faces clearly say "different person"
    "fused_weight_face_confirmed": 0.70,    # face dominates a confirmed match
    "fused_weight_appearance_confirmed": 0.30,
    "fused_weight_face_ambiguous": 0.50,    # balanced when the face is ambiguous
    "fused_weight_appearance_ambiguous": 0.50,
}
