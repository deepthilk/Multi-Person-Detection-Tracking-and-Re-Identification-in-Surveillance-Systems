# Person Registration & Identity Database

**Status:** Phase 2 module — independent, no changes required in
`detection/`, `tracking/`, `reidentification/`, `multicamera/`, or `web/`.

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

# Register one person
python register.py add --name "Alice" --images photos/alice_1.jpg photos/alice_2.jpg

# Register everyone at once (folder-of-folders: known_persons/<name>/*.jpg)
python register.py bulk --dir known_persons

# List / search
python register.py list
python register.py search --name ali

# Run the fast unit tests (no model download required)
python -m registration.tests.test_identity_db
```

## Why this design avoids merge conflicts

- New folder (`registration/`) + new root script (`register.py`) — no
  existing file is modified.
- Own `db_config.py` instead of touching the shared `config.py`.
- Imports `ReIDEngine` as-is; does not modify `reidentification/`.
- Does not import anything from `detection/`, `tracking/`, `multicamera/`,
  or `web/`.
- Owns `outputs/registration/` exclusively — no other module writes there.
