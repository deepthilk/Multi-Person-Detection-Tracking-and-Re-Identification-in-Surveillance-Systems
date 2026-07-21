"""
Final Integration: Multi-Camera -> Per-Camera Re-ID -> Cross-Camera Matching
                    -> Name Resolution -> single dashboard-ready JSON
================================================================================

This is the piece that ties every teammate's module together for Phase 3,
without editing any of their files:

  Lekha   (multicamera.multi_cam_pipeline)   — captures + tracks every camera,
                                                writes per-camera tracking.json
                                                in the exact schema reid_main.py
                                                already expects.
  Deepthi (reidentification.reid_main)       — run_reid_pipeline(), called
                                                ONCE PER CAMERA, unmodified.
  Deepthi (reidentification.cross_camera_match) — NEW: merges each camera's
                                                independent stable_ids into
                                                one global_id per real person.
  Prajna  (registration.identity_db)         — export_for_reid() supplies the
                                                {name: embedding} dict used to
                                                attach real names to global ids.
  Pranjali (web/ dashboard)                  — consumes the combined JSON this
                                                script writes.

Usage:
    python run_integrated_pipeline.py --device cuda
    python run_integrated_pipeline.py --device cpu --max-frames 200   # smoke test
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Full multi-camera Detection->Tracking->Re-ID->Cross-Camera pipeline")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--conf-threshold", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Optional cap on multicam ticks (quick smoke test)")
    parser.add_argument("--cross-cam-threshold", type=float, default=0.60,
                         help="Cosine similarity threshold for merging identities across cameras")
    parser.add_argument("--skip-names", action="store_true",
                         help="Skip name resolution even if the registration DB has entries")
    parser.add_argument("--output", type=str, default="outputs/cross_camera/global_identities.json")
    args = parser.parse_args()

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable — falling back to CPU")
        device = "cpu"

    # ── Step 1: multi-camera capture, detection & tracking (Lekha's module) ──
    logger.info("=" * 70)
    logger.info("STEP 1: Multi-Camera Capture, Detection & Tracking")
    logger.info("=" * 70)
    from multicamera.multi_cam_pipeline import run_multicamera_pipeline
    from multicamera.camera_config import CAMERAS, MULTICAM_SETTINGS

    records, per_camera_tracking = run_multicamera_pipeline(
        device=device, conf_threshold=args.conf_threshold, max_frames=args.max_frames,
    )
    logger.info(f"✅ {len(records)} person records across {len(per_camera_tracking)} camera(s)")

    # ── Step 2: Re-ID, once per camera, using Deepthi's UNMODIFIED pipeline ──
    logger.info("=" * 70)
    logger.info("STEP 2: Per-Camera Re-Identification")
    logger.info("=" * 70)
    from reidentification.reid_main import run_reid_pipeline

    camera_results = {}
    camera_engines = {}
    for cfg in CAMERAS:
        cam_id = cfg["camera_id"]
        if cam_id not in per_camera_tracking or not per_camera_tracking[cam_id]:
            logger.warning(f"⏭️  {cam_id}: no tracking data, skipping Re-ID")
            continue

        tracking_json = f"{MULTICAM_SETTINGS['tracking_dir']}/{cam_id}_tracking.json"
        reid_output = f"outputs/reid/{cam_id}_reid_results.json"
        logger.info(f"— {cam_id} ({cfg['source']}) —")

        engine, results = run_reid_pipeline(
            video_path=cfg["source"],
            tracking_json_path=tracking_json,
            output_json_path=reid_output,
            device=device,
        )
        camera_engines[cam_id] = engine
        camera_results[cam_id] = results

    if not camera_engines:
        logger.error("❌ No camera produced Re-ID results — nothing to cross-match")
        return 1

    # ── Step 3: cross-camera identity matching (NEW — Deepthi's final piece) ─
    logger.info("=" * 70)
    logger.info("STEP 3: Cross-Camera Identity Matching")
    logger.info("=" * 70)
    from reidentification.cross_camera_match import run_cross_camera_matching

    registered_persons = None
    if not args.skip_names:
        try:
            from registration.identity_db import IdentityDatabase
            db = IdentityDatabase()
            if len(db):
                registered_persons = db.export_for_reid()
                logger.info(f"Loaded {len(registered_persons)} registered person(s) for name resolution")
            else:
                logger.info("Registration DB is empty — global identities will be unnamed")
        except Exception as e:
            logger.warning(f"⚠️  Could not load registration DB ({e}); continuing without names")

    combined = run_cross_camera_matching(
        camera_results=camera_results,
        camera_engines=camera_engines,
        registered_persons=registered_persons,
        match_threshold=args.cross_cam_threshold,
        output_json_path=args.output,
    )

    n_global = len(combined.get("global_identities", {}))
    n_named = sum(1 for v in combined["global_identities"].values() if "name" in v)
    logger.info("=" * 70)
    logger.info(f"✅ Pipeline complete: {n_global} global identities "
                f"({n_named} matched to a registered name)")
    logger.info(f"   Dashboard-ready output: {args.output}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
