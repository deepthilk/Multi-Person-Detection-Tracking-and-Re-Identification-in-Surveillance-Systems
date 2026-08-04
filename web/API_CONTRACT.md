# Re-ID Registration & Auto-Fix — API Contract

Backend: FastAPI (served by `web/server.py`).

- Base URL (local): `http://127.0.0.1:8000`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`
- All responses are JSON (`Content-Type: application/json`) except image endpoints.
- Errors return `HTTP 4xx/5xx` with body `{"detail": "<message>"}`.
- CORS is open (`*`) for local development.
- File uploads use `multipart/form-data`; everything else uses `application/json`.

---

## Person registration

### `POST /api/registration` — register a person from uploaded photos

Form fields (multipart/form-data):

| Field         | Type      | Required | Notes                                  |
|---------------|-----------|----------|----------------------------------------|
| `name`        | string    | yes      | Person name (unique key)               |
| `files`       | file[]    | yes      | One or more JPEG/PNG photos            |
| `overwrite`   | bool      | no       | `true` replaces existing photos, `false` (default) appends |
| `augmentations` | int    | no       | Augmentation count used to stabilize the embedding (default `5`) |

Response `200`:

```json
{
  "registered": true,
  "person": {
    "name": "TestPerson4",
    "num_images": 1,
    "registered_at": "2026-07-31T01:01:42.483325",
    "last_updated": "2026-07-31T01:01:42.483343"
  }
}
```

### `GET /api/registration` — list all persons

```json
{
  "persons": [
    {
      "name": "Alice",
      "num_images": 2,
      "registered_at": "2026-07-30T23:51:30.016383",
      "last_updated": "2026-07-30T23:51:47.503291"
    }
  ]
}
```

### `GET /api/registration/{name}` — person details

```json
{
  "name": "TestPerson4",
  "num_images": 1,
  "registered_at": "2026-07-31T01:01:42.483325",
  "last_updated": "2026-07-31T01:01:42.483343",
  "image_paths": ["..."],
  "match_threshold": 0.55
}
```

### `DELETE /api/registration/{name}` — remove a person

Response `200`: `{"deleted": true, "name": "TestPerson4"}`

---

## Low-confidence check (verify)

### `POST /api/registration/{name}/verify` — compare a test image against the registration

Form fields (multipart/form-data):

| Field         | Type    | Required | Notes                          |
|---------------|---------|----------|--------------------------------|
| `test_image`  | file    | yes      | A crop/frame from the video    |
| `augmentations` | int  | no       | default `5`                    |

Response `200`:

```json
{
  "name": "TestPerson4",
  "similarity": 0.47,
  "threshold": 0.55,
  "passed": false,
  "gap": 0.08,
  "per_photo": [
    {"photo": 1, "similarity": 0.47}
  ],
  "suggest_autofix": true
}
```

**Frontend logic:** when `suggest_autofix` is `true` (or `passed` is `false`), offer the Auto-Fix flow below.

---

## Auto-Fix (find the person in surveillance video)

The scan is slow (detection + embedding over ~20 sampled frames), so it runs
as a **background job**. Flow:

1. `POST /api/registration/{name}/autofix` → get `job_id`
2. Poll `GET /api/autofix/{job_id}` every ~1.5 s until `status` is `completed` or `error`
3. Show the candidate crop thumbnails (`crop_url`) in a gallery
4. `POST /api/registration/{name}/autofix/confirm` to re-register from the chosen crop

### `GET /api/videos` — list cameras to scan

```json
{
  "videos": [
    {"camera_id": "cam1", "source": "input/video1.mp4", "exists": true},
    {"camera_id": "cam2", "source": "input/video2.mp4", "exists": true},
    {"camera_id": "cam3", "source": "input/video3.mp4", "exists": true}
  ]
}
```

Use `source` values as the `videos` array in the auto-fix request.

### `POST /api/registration/{name}/autofix` — start the scan

JSON body:

```json
{
  "videos": ["input/video1.mp4", "input/video2.mp4"],
  "samples": 20,
  "top": 5,
  "augmentations": 5
}
```

| Field   | Type   | Default | Notes                                |
|---------|--------|---------|--------------------------------------|
| `videos`| array  | —       | Required; camera sources from `/api/videos` |
| `samples`| int   | 20      | Number of evenly-spaced frames to scan |
| `top`   | int    | 5       | How many best candidates to keep     |
| `augmentations` | int | 5 | Embedding stabilization for each candidate |

Response `200`: `{"job_id": "c5b956c799", "status": "queued"}`

### `GET /api/autofix/{job_id}` — poll job status

```json
{
  "job_id": "c5b956c799",
  "name": "TestPerson4",
  "status": "completed",
  "progress": 100,
  "message": "Found 3 candidate(s)",
  "threshold": 0.55,
  "candidates": [
    {
      "index": 0,
      "video": "video1",
      "frame": 67,
      "bbox": [300, 120, 380, 340],
      "similarity": 0.914,
      "above_threshold": true,
      "crop_url": "/api/autofix/c5b956c799/crop/0",
      "crop_path": "outputs/registration/_auto_fix/candidate_video1_67_300_120.jpg"
    }
  ]
}
```

- `status`: `queued` → `running` → `completed` | `error`
- `progress`: integer 0–100
- `bbox`: `[x1, y1, x2, y2]` in the original frame
- `above_threshold`: `true` if `similarity >= threshold` (a confident find)
- In the gallery, highlight candidates where `above_threshold` is `true`.

### `GET /api/autofix/{job_id}/crop/{index}` — candidate crop image

Returns the JPEG thumbnail. Use directly in `<img src="...">`.

### `POST /api/registration/{name}/autofix/confirm` — register from a chosen crop

JSON body:

```json
{
  "job_id": "c5b956c799",
  "index": 0,
  "augmentations": 5
}
```

Response `200`:

```json
{
  "registered": true,
  "name": "TestPerson4",
  "source_video": "video1",
  "frame": 67,
  "similarity": 0.914,
  "crop_url": "/api/autofix/c5b956c799/crop/0"
}
```

The chosen crop becomes the person's registration image (`overwrite = true`),
so call this only when the user confirms the crop is the right person.

---

## Existing endpoints (video pipeline — used by the current UI)

### `POST /api/process` — run detection → tracking → re-id on an uploaded video

Multipart field `file` (video). Returns `{"job_id": "..."}`.

**Name-resolution rule (face-primary):** if a face was ever detected on a
track, the face decides the name — the best frame whose similarity to a
registered person's average face clears `face_match_threshold` (0.40) names the
person, and if no face clears the bar the person stays `unknown`. The body
(`match_threshold`, 0.55) is only consulted when a track never showed a face at
all (e.g. back-to-camera). An inconclusive face is never overruled by a body
guess.

### `GET /api/progress/{job_id}` — poll pipeline progress

```json
{
  "status": "running",
  "percent": 55,
  "message": "Running tracking",
  "output_url": "/outputs/<job_id>_reid.mp4",
  "output_name": "<job_id>_reid.mp4"
}
```

### Session run storage & cleanup (multi-camera sessions)

Every completed camera run leaves artifacts in `web/outputs` (`<session>_<cam>_reid.json`, `_reid.mp4`, `_reid_track*_top*.jpg`, `_manifest.json`) and the uploaded source clip in `web/uploads`. To keep disk use bounded, `server.py`:

- deletes the `_detections.json` / `_tracking.json` intermediates for each camera as soon as its run finishes (the UI never reads them after completion);
- prunes to the **newest 5 sessions** after each completed run and at startup — everything (outputs **and** the uploaded source clip) for older sessions is removed;
- sweeps orphaned 10-hex-prefixed files in `web/outputs` at startup (legacy `/api/process` job outputs, crash leftovers).

All deletion is strictly **session-scoped**: only files named `{10-hex-session}_{...}` are ever matched. Registered people — `outputs/registration/identity_db.json`, `web/uploads/registration/`, `reg_*`/`verify_*` photos, `outputs/registration/images/` — are never touched.

#### `GET /api/storage` — disk usage per session

```json
{
  "sessions": [
    { "session_id": "77d2f62a4a", "size": 12539871, "files": 9 }
  ],
  "total_size": 12539871,
  "session_count": 1
}
```

#### `DELETE /api/sessions/{session_id}` — delete one completed run

Deletes that session's outputs and source upload, and removes it from the in-memory index. Returns `409` if any of its cameras are still `running`/`queued`.

```json
{ "deleted": "77d2f62a4a", "removed_files": 9 }
```

#### `DELETE /api/sessions` — delete all completed runs

Skips sessions with cameras still `running`/`queued`. Registered people are unaffected.

```json
{ "deleted": ["77d2f62a4a", "25852b24aa"], "removed_files": 18 }
```

---

## Multi-person auto-fix (ALL registered persons, one scan)

Use this when a session came back with poor matches / "Unknown" people. One
scan pass over the videos produces the top-5 candidate crops for **every**
registered person at once, then each person is re-registered from the crop
the user picks. Same background-job pattern as above.

### `POST /api/autofix/all` — start the batch scan

JSON body:

```json
{
  "videos": ["input/video1.mp4"],
  "samples": 20,
  "top": 5,
  "augmentations": 5
}
```

Response `200`: `{"job_id": "cbdb7550c8", "status": "queued"}`

### `GET /api/autofix/{job_id}` — poll (batch shape)

When `status == "completed"` the job contains a `persons` array:

```json
{
  "job_id": "cbdb7550c8",
  "mode": "all",
  "status": "completed",
  "progress": 100,
  "message": "Top 5 candidate(s) for 3 person(s)",
  "threshold": 0.55,
  "persons": [
    {
      "name": "Alice",
      "candidates": [
        {
          "index": 0,
          "pool_index": 1,
          "video": "video1",
          "frame": 67,
          "bbox": [300, 120, 380, 340],
          "similarity": 0.914,
          "above_threshold": true,
          "crop_url": "/api/autofix/cbdb7550c8/crop/1",
          "crop_path": "outputs/registration/_auto_fix/candidate_video1_67_300_120.jpg"
        }
      ]
    }
  ]
}
```

- `candidates` are sorted best-first; render them as a **horizontal row of
  thumbnails** using `crop_url`.
- Highlight the ones where `above_threshold` is `true` (a confident find).
- `pool_index` is the shared crop pool — the same crop can appear for several
  persons; the `crop_url` always works via the pool index.

### `POST /api/autofix/all/confirm` — re-register a picked crop

JSON body:

```json
{
  "job_id": "cbdb7550c8",
  "name": "Alice",
  "index": 0,
  "augmentations": 5
}
```

Response `200`:

```json
{
  "registered": true,
  "name": "Alice",
  "source_video": "video1",
  "frame": 67,
  "similarity": 0.914,
  "crop_url": "/api/autofix/cbdb7550c8/crop/1"
}
```

Call this once per person after the user clicks a crop. The crop becomes that
person's registration image (`overwrite = true`).

---

## Target sightings / alerts

After a session finishes, every tracked person that resolved to a registered
name becomes a sighting. Alerts are **throttled per (name, camera)** for 5
minutes, so a person standing in front of a camera doesn't spam the feed.

### `GET /api/alerts`

```json
{
  "alerts": [
    {
      "id": "a1827be925",
      "timestamp": "2026-08-01T15:30:12",
      "session_id": "sess_abc123",
      "camera": "cam1",
      "camera_label": "Camera 1",
      "similarity": 0.91,
      "first_seen_sec": 3.2,
      "last_seen_sec": 12.7,
      "person": {
        "name": "Alice",
        "num_images": 1,
        "registered_at": "2026-07-30T23:51:30.016383",
        "last_updated": "2026-07-30T23:51:47.503291",
        "photos": ["/reg-photos/Alice/000_alice_1.jpg"]
      }
    }
  ],
  "count": 1
}
```

Poll this to drive the notification UI. `person.photos` are ready for
`<img src="...">`.

### `DELETE /api/alerts` — clear the feed

Response: `{"cleared": <number removed>}`

---

## Suggested UI workflow

1. **Register tab**: upload photo(s) + name → `POST /api/registration` → refresh list.
2. **Low-confidence tab**: upload a test frame → `POST /api/registration/{name}/verify`.
   - If `passed` → registration is fine.
   - If `suggest_autofix` → button "Find person in video".
3. **Auto-Fix tab**:
   - Load cameras via `GET /api/videos` (multi-select checkboxes).
   - `POST /api/registration/{name}/autofix` with selected sources.
   - Poll; render gallery of `crop_url` thumbnails with similarity badges.
   - On user click → `POST .../autofix/confirm` → show success, refresh person list.
4. **Multi-person auto-fix** (after a session with Unknowns):
   - `POST /api/autofix/all` with the session's videos.
   - Poll `GET /api/autofix/{job_id}`; for each person show a horizontal row
     of up to 5 `crop_url` thumbnails.
   - User clicks a crop per person → `POST /api/autofix/all/confirm` each →
     refresh results / re-run the session to see names.

## Gotchas

- `confirm` **overwrites** the person's registration image — the crop is used as the
  new photo. Only send it after the user picks a crop.
- Candidate crops persist under `outputs/registration/_auto_fix/`; they never overwrite
  each other (filenames include video + frame + bbox).
- All three demo videos are identical footage simulating three CCTV cameras — picking
  a crop from any of them is equivalent.
