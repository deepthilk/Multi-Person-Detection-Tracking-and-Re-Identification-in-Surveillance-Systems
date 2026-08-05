"""Run Re-ID only (skip detection/tracking) for v1, v2, v3."""
import sys, json, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reidentification.reid_main import run_reid_pipeline

for vname in ['v1', 'v2', 'v3']:
    video_path = f"input/{vname}.mp4"
    track_json = f"outputs/runs/{vname}/tracking.json"
    reid_json  = f"outputs/runs/{vname}/reid_results.json"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  Re-ID: {vname}.mp4")
    logger.info(f"{'='*60}")
    
    engine, results = run_reid_pipeline(
        video_path=video_path,
        tracking_json_path=track_json,
        output_json_path=reid_json,
        device="cpu",
    )
    
    ids = set()
    for fid, people in results.items():
        for p in people:
            cid = p.get('consolidated_id')
            if cid is not None and cid != -1:
                ids.add(cid)
    logger.info(f"  {vname}: {len(ids)} identities -> {sorted(ids)}")
