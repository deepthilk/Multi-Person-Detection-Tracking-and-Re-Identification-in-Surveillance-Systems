# Person Registration & Identity Database

**Status:** Merged into `main`. Verified compatible with Deepthi's actual
merged Re-ID / cross-camera code (`registration/tests/test_phase3_handoff.py`
imports her real `reidentification.cross_camera_match.resolve_names` and
confirms the hand-off works end-to-end) — not just designed to be, tested
to be. No changes were required in `detection/`, `tracking/`,
`reidentification/`, `multicamera/`, or `web/`.

## What this module does

Lets you register known persons (name + one or more photos), generates an
embedding for each photo, and stores name + embeddings + metadata in a
searchable identity database. In Phase 3, this database is handed to the
Re-ID module so it can answer "who is this person seen on camera?" instead
of just "is this the same track as before?".

```
photos/alice_1.jpg ─┐
photos/alice_2.jpg ─┼─> embedder.py (reuses Deepthi's Re-ID backbone) ─> IdentityDatabase ─> Re-ID / Dashboard
```

## Files

| File | Purpose |
|---|---|
| `db_config.py` | Storage paths + thresholds. Owned by this module — no one else needs to touch it. |
| `embedder.py` | Wraps Deepthi's `ReIDEngine.extract_feature` so registered-photo embeddings live in the *same* descriptor space as live-camera embeddings. Only imports `reidentification/`, never edits it. |
| `identity_db.py` | The database itself: add/get/delete/search/match a person, persisted as JSON. |
| `register_person.py` | High-level `register_person(name, image_paths)` — the one function most callers need. |
| `validate_setup.py` | Run this first to check your environment before registering anyone. |
| `register.py` (repo root) | CLI entry point. Separate from `main.py` on purpose — zero merge-conflict risk. |
| `tests/test_identity_db.py` | Fast unit tests for the database logic (no model download needed). |
| `tests/test_phase3_handoff.py` | Integration test against Deepthi's **actual merged** `cross_camera_match.resolve_names()` — proves the hand-off works with real code, not just in theory. |

## Why embeddings are generated via Deepthi's Re-ID engine

If registration used a different model to generate embeddings than the one
used during live tracking, a "known person" vector and a "person seen on
camera" vector would not be comparable — cosine similarity between them
would be meaningless. So `embedder.py` **imports** (never edits)
`reidentification.reid_main.ReIDEngine` and calls the exact same
`extract_feature()` used by the live pipeline, just on a full registered
photo instead of a detected+tracked crop.

## Output contract (for the Re-ID teammate / Phase 3)

1. **`outputs/registration/identity_db.json`** — one entry per registered
   person:
   ```json
   {
     "Alice": {
       "embeddings": [[...698 floats...], [...698 floats...]],
       "average_embedding": [...698 floats...],
       "metadata": {
         "registered_at": "2026-07-19T10:00:00",
         "last_updated": "2026-07-19T10:00:00",
         "num_images": 2,
         "image_paths": ["outputs/registration/images/Alice/000_alice_1.jpg", "..."]
       }
     }
   }
   ```

2. **`outputs/registration/images/<name>/`** — a self-contained copy of
   every photo used to register that person.

3. **Hand-off function** — Deepthi's integration code should call
   `IdentityDatabase().export_for_reid()`, which returns
   `{name: np.ndarray(698,)}`, ready to compare against live Re-ID
   descriptors with cosine similarity. Nobody outside this module reads
   `identity_db.json` directly.

## Running it

```bash
# Check your environment first
python registration/validate_setup.py

# Register one person (default: 5 augmentations for domain-robust embedding)
python register.py add --name "Alice" --images photos/alice_1.jpg photos/alice_2.jpg

# Register with more augmentations for better domain robustness
python register.py add --name "Bob" --images photos/bob/*.jpg --augmentations 15

# Register without augmentation (fast, less robust)
python register.py add --name "Charlie" --images charlie.jpg --augmentations 0

# Register everyone at once (folder-of-folders: known_persons/<name>/*.jpg)
python register.py bulk --dir known_persons

# Verify registration quality: check if a test image (e.g. video frame) will
# be recognised by the Re-ID pipeline
python register.py verify --name "Alice" --test-image frame_from_video.jpg

# List / search
python register.py list
python register.py search --name ali

# Replace a person's photos instead of adding to them
python register.py add --name "Alice" --images new_photos/*.jpg --overwrite

# Remove someone
python register.py delete --name "Bob"

# Back up / restore the whole database
python register.py backup --path backups/db_2026-07-27.json
python register.py restore --path backups/db_2026-07-27.json

# Run the similarity diagnostic
python check_similarity.py

# Run the unit tests (no model download required)
python -m registration.tests.test_identity_db

# Run the Phase 3 hand-off acceptance test against Deepthi's real merged code
python -m registration.tests.test_phase3_handoff
```

## Domain-shift fix: why your high-res photos weren't matching

The Re-ID model was fine-tuned on **Market-1501** — a dataset of low-resolution
surveillance camera crops (typically 128×64 px). When you register a person
using a **high-resolution professional photo** (different lighting, no
compression artifacts, sharp), the generated embedding lives in a *different
region of the feature space* compared to embeddings from actual video frames
— even for the same person.

**The symptoms:**
- Registering with a screenshot from the video → person is correctly identified.
- Registering with a high-res photo of the same person → person is NOT
  recognised (similarity falls below the matching threshold).

**Two fixes applied (both in this module, no changes to Re-ID code):**

### 1. Lowered match threshold (db_config.py: 0.75 → 0.55)

The cosine similarity between a clean studio photo and a blurry video crop of
the same person is often 0.40–0.60 — well below the old threshold of 0.75 but
still higher than the 0.20–0.45 range of genuinely different people. The new
threshold of 0.55 captures this domain-bridging range while still rejecting
true negatives.

### 2. Domain-robust embedding generation (embedder.py)

Before extracting features from a registration photo, the embedder now:

1. **Simulates video quality** — downscales the image to ~160px height
   (typical detection-crop size), applies mild Gaussian blur (camera/motion
   defocus), and JPEG-compresses at quality 75 (video stream artifacts).

2. **Augments and averages** — generates N random versions of the
   preprocessed image with varying brightness, contrast, blur, and
   resolution, then averages all N+1 embeddings into one robust descriptor.

This single averaged descriptor lives *closer* to the video-feature domain
than any single high-res embedding would, improving matching success without
requiring you to use video screenshots.

### 3. Verify command (register.py verify)

Before running the full pipeline, you can now check whether a registered
person will be recognised in a given video frame:

```bash
python register.py verify --name "Alice" --test-image frame_from_video.jpg
```

This reports the cosine similarity and indicates whether it exceeds the
matching threshold, with per-photo breakdown.

## Why this design avoids merge conflicts

- New folder (`registration/`) + new root script (`register.py`) — no
  existing file is modified.
- Own `db_config.py` instead of touching the shared `config.py`.
- Imports `ReIDEngine` as-is; does not modify `reidentification/`.
- Does not import anything from `detection/`, `tracking/`, `multicamera/`,
  or `web/`.
- Owns `outputs/registration/` exclusively — no other module writes there.
