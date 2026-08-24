# Project Guidelines

## Hard constraints (must always hold)

1. **Never reduce model accuracy.** Every change to the pipeline must either
   produce *numerically identical* model outputs, or pass an explicit
   before/after verification gate (compare: number of stable identities,
   matched vs unknown counts, and box IoU >= 0.9 on sampled frames) before it
   may be enabled by default. See `PERFORMANCE_REPORT.md` §7.
2. **Never change the 698-dim feature space** (descriptor composition,
   normalisation, or thresholds). It is the single invariant shared by
   registration, live Re-ID, and cross-camera matching.
3. **Never rewrite `reidentification/reid_main.py` internals or
   `finalize_clustering`** — they are correctness-critical. Prefer additive
   code (new params/methods, defaults unchanged).
4. Do not enable excluded tradeoffs (detection stride > 1, conf > 0.6, face
   det_size 320-primary, YOLOv8n, FP16/TensorRT/OpenVINO). See
   `PERFORMANCE_REPORT.md` §9.

## Environment

- Windows / PowerShell. Python 3.13 venv at `venv\Scripts\python.exe`.
- No CUDA on the dev machine — CPU only.
- Run scripts with `venv\Scripts\python.exe <script>.py`.
