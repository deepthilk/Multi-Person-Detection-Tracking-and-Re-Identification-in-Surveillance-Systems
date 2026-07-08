"""
Main orchestration script for the multi-person tracking and Re-ID system
Runs: Detection → Tracking → Re-Identification

CHANGES FROM ORIGINAL:
  run_tracking() now imports from tracking.track_module instead of defining
  DeepSORT inline. This means changes to track_module.py take effect here.

  DeepSORT settings (in track_module.py):
    max_age:             30 → 5   (prevents off-screen ghost bbox extrapolation)
    n_init:               3 → 2   (faster track confirmation)
    max_cosine_distance: 0.2 → 0.3 (better occlusion handling)
    bbox clamping: added before saving to JSON
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_detection(video_path, output_path, conf_threshold=0.35, device='cuda'):
    """Run YOLOv8 person detection"""
    logger.info("=" * 60)
    logger.info("STEP 1: Running Person Detection (YOLOv8)")
    logger.info("=" * 60)

    import cv2
    import json
    import torch
    from ultralytics import YOLO

    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("⚠️  CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'

    model_path = "models/yolov8s.pt"
    logger.info(f"Using device: {device}")
    logger.info(f"Loading YOLOv8 model from {model_path}")

    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Failed to open video: {video_path}")
        return False

    frame_id        = 0
    all_detections  = {}
    min_area        = 900
    min_height      = 50
    min_aspect      = 1.0
    max_aspect      = 4.5
    min_area_ratio  = 0.0008

    logger.info("Processing video frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        results    = model(frame, conf=conf_threshold, imgsz=960, device=device)[0]
        detections = []

        if results.boxes is not None:
            frame_h, frame_w = frame.shape[:2]
            min_area_dynamic = max(min_area, int(frame_w * frame_h * min_area_ratio))
            boxes   = results.boxes.xyxy.cpu().numpy()
            scores  = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                if int(cls) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1; h = y2 - y1
                area   = w * h
                aspect = h / max(w, 1)

                if area < min_area_dynamic or h < min_height:
                    continue
                if aspect < min_aspect or aspect > max_aspect:
                    continue
                detections.append([x1, y1, w, h, float(score)])

        all_detections[frame_id] = detections

        if frame_id % 100 == 0:
            logger.info(f"Processed {frame_id} frames, detected {len(detections)} persons")

    cap.release()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(output_path, 'w') as f:
        json.dump(all_detections, f, indent=4)

    logger.info(f"✅ Detection completed")
    logger.info(f"   Total frames: {frame_id}")
    logger.info(f"   Total detections: {sum(len(d) for d in all_detections.values())}")
    logger.info(f"   Output: {output_path}")
    return True


def run_tracking(video_path, detection_path, output_path):
    """
    Run DeepSORT tracking.
    Delegates entirely to tracking.track_module so settings are in one place.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Running Person Tracking (DeepSORT)")
    logger.info("=" * 60)

    # FIX: import from track_module instead of inline DeepSort with hardcoded settings.
    # This means editing tracking/track_module.py is sufficient to change tracker behaviour.
    from tracking.track_module import run_tracking as _run_tracking

    result = _run_tracking(video_path, detection_path, output_path)

    if result is None:
        logger.error("❌ Tracking failed")
        return False

    logger.info(f"✅ Tracking completed  Output: {output_path}")
    return True


def run_reid(video_path, tracking_path, output_path, device='cuda'):
    """Run Re-Identification"""
    logger.info("=" * 60)
    logger.info("STEP 3: Running Re-Identification (OSNet)")
    logger.info("=" * 60)

    from reidentification.reid_main import run_reid_pipeline

    try:
        run_reid_pipeline(
            video_path          = video_path,
            tracking_json_path  = tracking_path,
            output_json_path    = output_path,
            device              = device,
        )
        logger.info(f"✅ Re-ID completed  Output: {output_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Re-ID failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Multi-person tracking and Re-ID system'
    )
    parser.add_argument('--video', type=str, default='input/video4.mp4',
                        help='Input video path')
    parser.add_argument('--step', type=int, default=3, choices=[1, 2, 3],
                        help='Run up to step: 1=detection, 2=tracking, 3=reid')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--conf-threshold', type=float, default=0.35,
                        help='Detection confidence threshold')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results after pipeline')
    parser.add_argument('--skip-detection', action='store_true',
                        help='Skip detection (reuse existing detections.json)')
    parser.add_argument('--skip-tracking', action='store_true',
                        help='Skip tracking (reuse existing tracking.json)')
    parser.add_argument('--skip-reid', action='store_true',
                        help='Skip Re-ID')

    args = parser.parse_args()

    video_path        = args.video
    detection_output  = 'outputs/detections.json'
    tracking_output   = 'outputs/tracking.json'
    reid_output       = 'outputs/reid_results.json'

    logger.info("Multi-Person Tracking & Re-ID System")
    logger.info("=" * 60)
    logger.info(f"Video : {video_path}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    if not Path(video_path).exists():
        logger.error(f"❌ Video not found: {video_path}")
        return 1

    # Step 1: Detection
    if args.step >= 1 and not args.skip_detection:
        if not run_detection(video_path, detection_output,
                             args.conf_threshold, args.device):
            return 1
    else:
        logger.info("⏭️  Skipping detection")

    # Step 2: Tracking
    if args.step >= 2 and not args.skip_tracking:
        if not run_tracking(video_path, detection_output, tracking_output):
            return 1
    else:
        logger.info("⏭️  Skipping tracking")

    # Step 3: Re-ID
    if args.step >= 3 and not args.skip_reid:
        if not run_reid(video_path, tracking_output, reid_output, args.device):
            logger.warning("⚠️  Re-ID failed, but previous steps completed")
    else:
        logger.info("⏭️  Skipping Re-ID")

    logger.info("=" * 60)
    logger.info("✅ Pipeline completed!")
    logger.info("=" * 60)

    if args.visualize:
        from utils import visualize_results
        logger.info("Visualizing tracking results...")
        visualize_results(
            video_path     = video_path,
            detections_json = detection_output,
            tracking_json  = tracking_output,
            mode           = 'tracking',
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())