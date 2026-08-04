"""
Person Detection Module (YOLOv8)
Detects persons in video frames
"""

import cv2
import json
import torch
from ultralytics import YOLO
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PersonDetector:
    """YOLOv8 based person detector"""

    def __init__(
        self,
        model_path='models/yolov8s.pt',
        conf_threshold=0.25,
        device='cuda',
        min_area=600,
        min_height=30,
        min_aspect=0.5,
        max_aspect=4.5,
        min_area_ratio=0.0008,
        weak_conf_threshold=None,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        # Detections at [weak_conf_threshold, conf_threshold) are "barely
        # visible" people: kept only when they don't overlap a strong (>= conf)
        # detection, so the main gate stays tight while edge/occluded people
        # still get a chance to be tracked and matched.
        self.weak_conf_threshold = (
            weak_conf_threshold if weak_conf_threshold is not None else conf_threshold
        )
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.min_area = min_area
        self.min_height = min_height
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area_ratio = min_area_ratio
        
        logger.info(f"Loading YOLOv8 model from {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        logger.info(f"✅ Detector initialized on {self.device}")
    
    def detect(self, frame, imgsz=960):
        """
        Detect persons in frame
        
        Args:
            frame: OpenCV frame
            imgsz: Input image size for YOLO
        
        Returns:
            List of [x1, y1, w, h, score] detections
        """
        run_conf = min(self.conf_threshold, self.weak_conf_threshold)
        results = self.model(frame, conf=run_conf, imgsz=imgsz, device=self.device)[0]
        strong = []
        weak = []
        
        if results.boxes is not None:
            frame_h, frame_w = frame.shape[:2]
            min_area_dynamic = max(self.min_area, int(frame_w * frame_h * self.min_area_ratio))
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                if int(cls) != 0:  # Only keep person class (class 0)
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1
                area = w * h
                aspect = h / max(w, 1)

                if area < min_area_dynamic or h < self.min_height:
                    continue

                if aspect < self.min_aspect or aspect > self.max_aspect:
                    continue

                det = [x1, y1, w, h, float(score)]
                if score >= self.conf_threshold:
                    strong.append((box, det))
                elif score >= self.weak_conf_threshold:
                    weak.append((box, det))
        
        # Drop weak boxes that overlap a strong one (same person, fluctuating
        # confidence) — only genuinely separate barely-visible people survive.
        kept_weak = []
        for wbox, det in weak:
            if any(_iou(wbox, sbox) > 0.5 for sbox, _ in strong):
                continue
            kept_weak.append(det)
        
        return [det for _, det in strong] + kept_weak


def _iou(a, b):
    """Intersection over union for two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    b_area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(a_area + b_area - inter)


def run_detection(
    video_path,
    output_path,
    conf_threshold=0.25,
    imgsz=960,
    device='cuda',
    min_area=600,
    min_height=30,
    min_aspect=0.5,
    max_aspect=4.5,
    min_area_ratio=0.0008,
    weak_conf_threshold=None,
):
    """
    Run person detection on entire video
    
    Args:
        video_path: Input video path
        output_path: Output JSON path
        conf_threshold: Detection confidence threshold (strong tier)
        weak_conf_threshold: Lower bound for "barely visible" people; kept only
            when they don't overlap a strong detection
        imgsz: YOLO input image size
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary of frame_id -> detections
    """
    detector = PersonDetector(
        conf_threshold=conf_threshold,
        device=device,
        min_area=min_area,
        min_height=min_height,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        min_area_ratio=min_area_ratio,
        weak_conf_threshold=weak_conf_threshold,
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None
    
    frame_id = 0
    all_detections = {}
    
    logger.info(f"Processing video: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        detections = detector.detect(frame, imgsz=imgsz)
        all_detections[frame_id] = detections
        
        if frame_id % 50 == 0:
            logger.info(f"Frame {frame_id}: {len(detections)} detections")
    
    cap.release()
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_detections, f, indent=4)
    
    logger.info(f"✅ Detection complete: {frame_id} frames, {sum(len(d) for d in all_detections.values())} total detections")
    
    return all_detections


if __name__ == "__main__":
    import sys
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else "input/video4.mp4"
    output_path = "outputs/detections.json"
    
    run_detection(video_path, output_path)
