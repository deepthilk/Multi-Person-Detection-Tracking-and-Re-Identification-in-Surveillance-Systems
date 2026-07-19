"""
Standalone entry point for the Multi-Camera Processing module.

Kept SEPARATE from main.py on purpose: main.py is the single-video
Detection->Tracking->Re-ID pipeline other teammates already work on, so a
new top-level script avoids any merge conflicts there. Once every module
is ready, integration can either import run_multicamera_pipeline() from
main.py, or main.py can be extended in a dedicated integration branch.

Usage:
    python run_multicam.py
    python run_multicam.py --device cuda --conf-threshold 0.4
    python run_multicam.py --max-frames 200      # quick smoke test
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Camera Processing and Video Stream Management"
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--conf-threshold", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Optional cap on ticks processed (useful for quick tests)")
    args = parser.parse_args()

    from multicamera.multi_cam_pipeline import run_multicamera_pipeline

    logger.info("=" * 60)
    logger.info("Multi-Camera Processing Module")
    logger.info("=" * 60)

    records, per_camera_tracking = run_multicamera_pipeline(
        device=args.device,
        conf_threshold=args.conf_threshold,
        max_frames=args.max_frames,
    )

    logger.info("=" * 60)
    logger.info(f"✅ Done. {len(records)} standardized person records produced "
                f"across {len(per_camera_tracking)} camera(s).")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
