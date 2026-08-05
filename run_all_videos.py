"""
Run the full Detection -> Tracking -> ReID pipeline on multiple videos.
Each video gets its own output folder under outputs/runs/<video_name>/
"""
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve()))


def run_all(videos, device="cpu"):
    from detection.detect_module import run_detection
    from tracking.track_module import run_tracking
    from reidentification.reid_main import run_reid_pipeline

    results_summary = {}

    for vname in videos:
        video_path = f"input/{vname}.mp4"
        if not Path(video_path).exists():
            logger.error(f"Video not found: {video_path}")
            continue

        out_dir = Path(f"outputs/runs/{vname}")
        out_dir.mkdir(parents=True, exist_ok=True)
        det_json = str(out_dir / "detections.json")
        track_json = str(out_dir / "tracking.json")
        reid_json = str(out_dir / "reid_results.json")

        logger.info(f"\n{'='*60}")
        logger.info(f"  PROCESSING: {vname}.mp4")
        logger.info(f"{'='*60}")

        # Step 1: Detection
        logger.info(f"\n--- Step 1/3: Detection ---")
        try:
            run_detection(video_path, det_json, device=device)
        except Exception as e:
            logger.error(f"  Detection failed: {e}")
            import traceback; traceback.print_exc()
            continue

        # Step 2: Tracking
        logger.info(f"\n--- Step 2/3: Tracking ---")
        try:
            run_tracking(video_path, det_json, track_json)
        except Exception as e:
            logger.error(f"  Tracking failed: {e}")
            import traceback; traceback.print_exc()
            continue

        # Step 3: ReID
        logger.info(f"\n--- Step 3/3: ReID ---")
        try:
            engine, results = run_reid_pipeline(
                video_path=video_path,
                tracking_json_path=track_json,
                output_json_path=reid_json,
                device=device,
            )
            all_ids = set()
            for fid, people in results.items():
                for p in people:
                    cid = p.get('consolidated_id')
                    if cid is not None and cid != -1:
                        all_ids.add(cid)
            logger.info(f"  ReID done: {len(all_ids)} unique identities -> {reid_json}")
            results_summary[vname] = {
                "video": video_path,
                "frames": len(results),
                "identities": sorted(all_ids),
                "num_identities": len(all_ids),
                "reid_output": reid_json,
            }
        except Exception as e:
            logger.error(f"  ReID failed: {e}")
            import traceback; traceback.print_exc()
            continue

    # Print summary
    logger.info(f"\n\n{'='*60}")
    logger.info(f"  SUMMARY")
    logger.info(f"{'='*60}")
    for vname, info in results_summary.items():
        logger.info(f"  {vname}: {info['num_identities']} identities "
                     f"(IDs: {info['identities']}) -> {info['reid_output']}")
    logger.info(f"{'='*60}")

    return results_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run pipeline on multiple videos")
    parser.add_argument("--videos", "-v", type=str, default="v1,v2,v3",
                        help="Comma-separated video names (without extension)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"],
                        help="Device to use")
    args = parser.parse_args()
    videos = [v.strip() for v in args.videos.split(",")]
    device = args.device
    run_all(videos, device=device)
