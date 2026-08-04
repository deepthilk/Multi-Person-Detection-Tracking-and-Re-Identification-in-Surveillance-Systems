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
    "seed": 42,               # Fixed RNG seed for the video-quality simulation
                               # and random augmentation, so registering the
                               # same photo always produces the SAME embedding.
                               # Deterministic registration means re-running
                               # the demo yields identical results every time
                               # (a deployment-minded reviewer will re-run it).
}

# ==============================================================================
# SEARCH / MATCH SETTINGS
# ==============================================================================

SEARCH_SETTINGS = {
    # Cosine similarity threshold used by IdentityDatabase.match() when
    # comparing a query embedding (e.g. from a live Re-ID track) against
    # registered persons.
    #
    # LOWERED from 0.75 → 0.55 to accommodate domain shift between
    # high-resolution registration photos and low-resolution surveillance
    # video crops. The Re-ID model's embeddings vary significantly with
    # image quality, lighting, and compression — a threshold that works
    # for same-domain matching (e.g. Market-1501 query/gallery) is too
    # strict when comparing a clean studio photo to a blurry video frame
    # of the same person. See embedder.py's video-simulation preprocessing
    # for the complementary fix on the embedding side.
    "match_threshold": 0.55,
    "top_k": 3,

    # Face-aware matching bars (SFace/ONNX cosine scale, see face_cue.py):
    #   * Above `face_match_threshold` a visible face CONFIRMS the identity on
    #     its own — a strong face beats a weak body score, which is exactly
    #     the "same person, different outfit/lighting" case. SFace's own
    #     same-person cutoff is ~0.363; live different-people peaks here
    #     stayed under 0.33, so 0.40 sits safely between them.
    #   * Below `face_veto_threshold` a confident face is a MISMATCH — two
    #     different people can't share a face, so it vetoes even a strong
    #     body-appearance score.
    # Between the two the match falls back to the soft 0.75*face + 0.25*body
    # blend against `match_threshold`.
    "face_match_threshold": 0.40,
    "face_veto_threshold": 0.30,
}

# ==============================================================================
# GUARDRAIL SETTINGS (registration-time sanity checks)
# ==============================================================================

GUARDRAIL_SETTINGS = {
    # Guardrails only WARN — they never block a registration. A system that
    # auto-refused would annoy operators; one that flags the risk in the
    # console and the self-check report keeps the risk visible and auditable.
    #
    # Body-descriptor cross-similarity at or above this means the new person's
    # average appearance embedding is effectively indistinguishable from an
    # existing registrant's. Measured live on this project's uniformed
    # subjects, DIFFERENT people scored 0.90–0.96 on the body cue alone, so a
    # registrant pair above 0.85 cannot be separated by appearance.
    "body_discrimination_threshold": 0.85,

    # Face-descriptor cross-similarity at or above this is a red flag: either
    # the same person is registered twice under different names, or the photo
    # set is unusable. Measured live, DIFFERENT people scored <= 0.29 on the
    # face cue, so 0.50 leaves a wide safety margin.
    "face_discrimination_threshold": 0.50,

    # Directory for timestamped full-DB backups taken automatically before any
    # destructive operation (register_person with overwrite=True). Every such
    # operation can be rolled back via IdentityDatabase.import_db().
    "backups_dir": "outputs/registration/backups",
}
