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
    # Body-only match threshold (cosine of 698-dim body descriptors).
    #
    # Must be ABOVE the inter-person body similarity range (0.90–0.96 for
    # uniformed subjects, per guardrail measurements) to prevent false
    # positives between different people wearing similar outfits.  At 0.85,
    # same-person cross-domain matches (0.90+) still pass, while different-
    # person matches (0.90–0.96) are correctly rejected.  If face detection
    # succeeds, this threshold is irrelevant — face decides at 0.40.
    "match_threshold": 0.85,
    "top_k": 3,

    # Face-aware matching bars (ArcFace/ONNX cosine scale, see face_cue.py):
    #   * Above `face_match_threshold` a visible face CONFIRMS the identity on
    #     its own — a strong face beats a weak body score, which is exactly
    #     the "same person, different outfit/lighting" case. ArcFace-R100's
    #     same-person cosine is typically 0.4–0.7; different-people peaks
    #     around 0.20–0.25, so 0.40 sits safely between them.
    #   * Below `face_veto_threshold` a confident face is a MISMATCH — two
    #     different people can't share a face, so it vetoes even a strong
    #     body-appearance score.
    # Between the two the match falls back to the body score against
    # `match_threshold`.
    "face_match_threshold": 0.45,
    "face_veto_threshold": 0.25,
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
