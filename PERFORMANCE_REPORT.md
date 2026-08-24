# Pipeline Performance & Real-Time Report

Multi-Person Detection, Tracking & Re-Identification in Surveillance Systems

**Guiding constraint: every technique listed below either produces *numerically identical* model outputs, or carries an explicit verification gate proving model accuracy is unchanged. Techniques that can reduce model accuracy/efficiency are listed separately as *excluded* tradeoffs and are not recommended.**

---

## 1. Where the time actually goes (hotspot audit)

Ordered by estimated cost per pipeline run:

1. **Re-ID stage — dominant (CPU-bound)**
   - `ReIDEngine.process_frame` runs the ResNet-50 Re-ID backbone at 256×128 (`reidentification/reid_main.py:379`) and InsightFace ONNX face detection (`reidentification/reid_main.py:1182`) **per person, per frame**.
   - InsightFace runs on **CPU only** (`CPUExecutionProvider`, `det_size=(640, 640)` — `reidentification/insight_face.py:67-71`). Heads are usually small (top-40% of the bbox), so a 640² face-detection pass on a tiny crop is wasteful.
   - Loading the `buffalo_s` ONNX model pack costs **several seconds per instance** (`reidentification/insight_face.py:152`). `web/multicam_pipeline.py` can load **two** instances per camera job (the engine's own + `get_shared_extractor()`).

2. **Redundant full-video decodes** (`web/server.py`, `utils.py`)
   - A web camera job decodes the whole video: detection pass → Re-ID pass → render pass (`utils.py render_reid_video`) → full ffmpeg libx264 re-encode (`utils.py _reencode_for_browser`). Then `_unify_session` **re-renders and re-encodes every camera again** — even cameras whose names did not change (`web/server.py:559-574`). That is roughly 3 decodes + 2 encodes per clip.

3. **Sequential multi-camera jobs**
   - FastAPI `BackgroundTasks` run one after another on a single event loop (`web/server.py:409-422`). A 3-camera session takes ~3× a single camera's wall-clock time, with zero overlap.

4. **Detection at full resolution**
   - `config.py:13` sets `imgsz=960`. The codebase's own comment (`detection/detect_module.py:171`) states **640 ≈ equal detection quality at 2.3× fewer pixels** on this project's footage.

5. **JSON I/O**
   - Every stage writes `indent=4` JSON (`detection/detect_module.py:221`, `reidentification/reid_main.py`, `web/multicam_pipeline.py:255`) — large per-frame dicts, slow to write and re-parse.

6. **Track gap-filling + thumbnails**
   - `fill_track_gaps` materialises every frame (necessary for the overlay), and `extract_track_thumbnail` opens + seeks the video once per track (`utils.py:399`).

---

## 2. Baseline assumptions (so estimates are meaningful)

All time estimates below are **estimates to be verified with the baseline profiler** (Phase 0). They assume:

- 60-second clip @ 30 fps (1,800 frames), 3–5 people on screen, **CPU-only**, single camera.
- Detection: ~340 ms/frame @ imgsz 960, ~150 ms/frame @ imgsz 640 (per the OpenVINO-vs-torch note in `detection/detect_module.py:43`).
- Re-ID, per person-frame: ~80–150 ms ResNet-50 forward + ~100–300 ms InsightFace ONNX @ 640.
- Render + ffmpeg re-encode: ~90 s per clip.

Real timings may differ; the profiler is the source of truth.

---

## 3. Time-savings table

| Ref | Technique | Est. time saved (1 cam / 1,800 frames / CPU) | Accuracy impact | Risk |
|---|---|---|---|---|
| A1 | Detection imgsz 640 (from 960) | ~5 min | **Gate** (≈equal per repo comment) | none |
| A3 | Re-ID stride 2–3 | ~3–10 min | **Gate** | none |
| A5 | CUDA where available | ~10–30 min | Identical | none |
| A6 | Compact JSON | ~30 s | Identical | none |
| B1 | Batched YOLO inference | ~1–3 min | Identical | low |
| B2 | Batched deep-feature extraction | ~2–6 min | Identical (eval-mode confirmed) | low |
| B3 | Quality-aware face throttle | ~4–10 min | **Gate** | low |
| B4 | Shared InsightFace extractor | ~5–15 s per camera job (startup) | Identical | none |
| B5 | Parallel web camera jobs | ~2–3× session wall-clock | Identical | low |
| B6 | Skip no-op re-renders + single encode pass | ~90–180 s per clip | Identical | low |
| B7 | Partial overlay streaming | N/A (UX) | Identical | none |
| R1 | Pseudo-live for uploads | N/A (UX) | Identical | none |
| R2 | Online live pipeline | 3–10 fps CPU / 25–30 fps GPU | Same models | med |
| R3 | Multi-RTSP live (R2 + per-cam workers) | N/A (UX) | Same models | med |

`**Gate**` = must pass the before/after verification checklist in §7 before it may be enabled by default.

---

## 4. Tier A — config/usage only (zero code, zero risk)

| # | Change | Where | Accuracy policy |
|---|---|---|---|
| A1 | Detection `imgsz` 640. Already the runtime default in `detection/detect_module.py` / `multicamera/multi_cam_pipeline.py`; stop overriding to 960 (`main.py:71`, `config.py:13`). | `main.py`, `config.py` | Gate — repo comment reports ≈equal quality on this footage |
| A3 | Re-ID stride 2–3. Identity results barely change across consecutive frames; runtime drops ~stride×. Params already exist in `run_reid_pipeline(stride=...)`, `run_camera_reid(stride=...)`, web session `stride`. | entry points / web | Gate — confirm identity count unchanged |
| A5 | Run on CUDA where available (web already auto-selects; CLI scripts default `cpu`). | CLI flags | Identical weights/behaviour |
| A6 | Compact JSON (`indent=None`) for large per-frame files; keep a readable option for debugging. | all writers | Identical |
| A7 | **EXCLUDED** — switching face `det_size` to 320-primary. The current 640-first + 320-retry logic (`insight_face.py:113`) already handles small faces; changing the default can change face-recall. | — | Excluded |

## 5. Tier B — small additive code (new params/flags, defaults unchanged)

| # | Change | Why it is accuracy-invariant | Est. win |
|---|---|---|---|
| B1 | **Batch YOLO inference** — `PersonDetector.detect()` accepts a frame list; accumulate N frames, one `model(frames, ...)` call. | ultralytics batched inference runs the same weights/preprocessing per image; outputs are identical. | 1.5–2× detection |
| B2 | **Batched deep-feature extraction** — new `ReIDEngine.extract_features_batch(frame, bboxes)`; collect all boxes in a frame, one backbone forward. | Model is already `eval()` at load (`reid_main.py:642`) → frozen BatchNorm running stats → batched forward is mathematically identical to per-image. | ~N× on ResNet part |
| B3 | **Quality-aware face throttle** — run InsightFace on a track at most every K frames, but *always* on a track's first sighting and whenever the head bbox is small or no confident face is stored yet. | Never starves the rare clear face (the failure mode `web/multicam_pipeline.py:119` documents); per-track face galleries are capped at 12 anyway. | ~K× on face calls |
| B4 | Reuse `get_shared_extractor()` inside `ReIDEngine._build_face_extractor` instead of building a fresh `InsightFaceExtractor`. | Same model, same extraction, one load instead of two. | seconds/job startup |
| B5 | **Parallel web camera jobs** — replace `BackgroundTasks` with a `ThreadPoolExecutor` (one thread per camera). Each camera owns its own detector/tracker/engine, so there is no shared mutable model state. | Per-camera pipeline unchanged; cross-camera unification still runs after all complete (as today). | ~N× session wall-clock |
| B6 | In `_unify_session`, skip re-render for cameras with **no name changes**, and ffmpeg-re-encode only the final version. | Only skips work whose output would be identical. | halves render+encode |
| B7 | Serve **partial overlay** from `/api/session/{id}/camera/{cid}/tracks` as frames complete. | Display-only; no model involved. | dashboard feels live |

## 6. Real-time approaches (new additive modules)

### R1 — "Pseudo-live" for uploaded clips (zero new infra)
The Processing tab already plays the uploaded video with a precomputed overlay (`web/static/app.js` `maybeFetchRealTracks`). Extend `/camera/{id}/tracks` to return boxes for frames processed so far and keep polling — the UI becomes live as the job runs.

### R2 — Online live pipeline (recommended real-time path)
New `live_pipeline.py` (mirrors `run_multicam.py`; zero edits to existing modules):
- `MultiCameraStreamManager` opens **file / webcam index / RTSP** (already supported by `multicamera/stream_manager.py`).
- Per tick: `PersonDetector.detect()` → `PersonTracker.update()` → `ReIDEngine.process_frame()` on a sampling basis.
- **Track-level identity throttling** (the `track_cluster.py` philosophy applied online): compute the 698-dim descriptor once per track per K frames; run face once per track (best face); resolve name per **track event** via `IdentityDatabase.match_multimodal` — not per frame.
- Emit `(camera_id, frame, boxes, ids, names)` via a WebSocket/SSE route in `web/server.py` (additive).
- Same models as today; accuracy profile matches the batch path's gate (stride).

### R3 — Full multi-camera RTSP live
R2 + one worker thread per camera + a rolling cross-camera unifier (reuse `CrossCameraMatcher` on a sliding window of track descriptors). Same building blocks, no rework.

---

## 7. How we guarantee no accuracy loss (verification checklist)

### Why the "identical" techniques are provably neutral
- **Same weights, same preprocessing, same thresholds** — the change is only *how many* images flow through the model per call (batching), *which* images are embedded (throttling), or *which hardware* runs the forward pass (CUDA vs CPU).
- **Frozen BatchNorm**: `ReIDEngine._load_model` calls `m.eval()` at load (`reidentification/reid_main.py:642`). Batched inference therefore produces the same embeddings as per-image inference — no training/eval-mode drift.
- **ultralytics batched inference** applies identical per-image preprocessing; per-image outputs are identical to the single-image path.

### Gate before enabling A1, A3, B3 by default
1. Run the current pipeline once (baseline) on each target clip.
2. Apply the technique.
3. Compare: **(a) number of stable identities**, **(b) matched vs unknown counts** (exposed by `/api/session/{id}/results` summary and `run_all_videos.py`), **(c) spot-check boxes on ~5 sampled frames** (IoU ≥ 0.9 vs baseline).
4. Enable only if (a) and (b) are unchanged and (c) passes.

---

## 8. Recommended rollout order

| Phase | Items | Risk | Wall-clock gain |
|---|---|---|---|
| 0 | Baseline profiler (`profile_pipeline.py`, additive, reuses the `LATENCY_SAMPLES` pattern from `web/server.py`) | none | — |
| 1 | A1 + A5 + A6 (imgsz 640, CUDA, compact JSON) | none | ~2–6× |
| 2 | A3 (Re-ID stride 2–3) **after gate passes** | none | another ~2–3× |
| 3 | B1 + B2 + B3 + B4 (batched YOLO, batched features, quality-aware face throttle, shared extractor) | low | another ~2–4× |
| 4 | B5 + B6 + B7 (parallel jobs, single render pass, partial overlay) | low | ~N× sessions |
| 5 | R1 → R2 (`live_pipeline.py` + WebSocket) → R3 | low–med | real-time UX |

---

## 9. Excluded (can reduce accuracy/efficiency — not recommended)

| Item | Why excluded |
|---|---|
| Detection stride >1 (A2) | `detect_module.py:171` states stale gap-frame boxes degrade downstream face extraction; the repo deliberately defaults to stride 1 for accuracy. |
| Raising confidence above current 0.6 in the web path | Cuts recall — low-confidence detections are dropped. |
| Face `det_size` 320 primary (A7) | Current 640-first + 320-retry already covers small faces; changing the default risks face-recall. |
| Lightweight online embedder (R4) | Introduces a second feature stream that is not comparable to the registration DB's 698-dim space — risks silent name-resolution failures. |
| YOLOv8n instead of v8s | Lower accuracy than the current detector. |
| FP16 / TensorRT / OpenVINO | FP16 is usually near-lossless but still a numerical change; TensorRT/OpenVINO were measured **10–20× slower** than torch on this laptop's CPU (`detect_module.py:43`) and are GPU-only anyway. |

---

## 10. Explicitly avoided (to keep the code unbroken)

- Changing the 698-dim descriptor composition or normalisation (breaks the single feature-space invariant across registration, live Re-ID, cross-camera matching).
- Retuning global matching thresholds (MATCH / REAPPEAR / FALLBACK / REACT).
- Rewriting `reidentification/reid_main.py` internals or `finalize_clustering` (correctness-critical).
- Sharing one model instance across threads/processes without per-camera isolation.
- Skipping `finalize_clustering` / `web/face_verify.py` for speed (they fix real false-merges).

---

## 11. Implementation status

Measured on this dev machine (CPU-only), `input/video1.mp4` (478×850 @ 29.99 fps, 346 frames).

| Ref | Status | Evidence |
|---|---|---|
| A6 | ✅ Done | Compact JSON (`separators=(",", ":")`) in `detection/detect_module.py`, `web/multicam_pipeline.py:255`, `web/face_verify.py:874`, `web/server.py`. reid.json ~38% smaller (1071 KB → 668 KB). |
| B1 | ✅ Done | `PersonDetector.detect_batch` + batched `run_detection` loop (`batch_size=8` default). Detection stage 265s → 90s (~3×). **Verified numerically identical: 346/346 frames matched** (boxes exactly equal). |
| B4 | ✅ Done | `get_shared_extractor()` singleton with creation lock; `InsightFaceExtractor` calls guarded by a per-instance lock for concurrent camera jobs. |
| B5 | ✅ Done | Camera jobs run on `threading.Thread`; `_join_threads_then_unify` preserves unify-after-all ordering. |
| B6 | ✅ Done | Job-level render removed; one render in `_unify_session` with final names; `_rewrite_tracks` returns whether labels changed; `_re_render` sets `output_url`; fallback render on unify failure. |
| A3 / B3 | ⏳ Gated | Re-ID stride / quality-aware face throttle — require the §7 before/after checklist before enabling by default. |
| B2 | 🚫 Not done | Would require touching `reid_main.py` internals (see §10). |

Constraints recorded in `AGENTS.md` (accuracy never drops; 698-dim invariant; no `reid_main.py` rewrites; excluded tradeoffs off-limits).
