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
        conf_threshold=0.35,
        device='cuda',
        min_area=900,
        min_height=50,
        min_aspect=1.0,
        max_aspect=4.5,
        min_area_ratio=0.0008,
        dedup_cover_ratio=0.9,
        edge_margin=2,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.min_area = min_area
        self.min_height = min_height
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area_ratio = min_area_ratio
        self.dedup_cover_ratio = dedup_cover_ratio
        self.edge_margin = edge_margin
        
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
        results = self.model(frame, conf=self.conf_threshold, imgsz=imgsz, device=self.device)[0]
        detections = []
        
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
                detections.append([x1, y1, w, h, float(score)])

        return self._clean_detections(detections, frame_w, frame_h)

    def _clean_detections(self, detections, frame_w, frame_h):
        """
        Post-process raw YOLO detections to remove duplicates and partials:
          1. Suppress a detection that is nearly fully contained in a larger one
             (duplicate boxes on the same person).
          2. Drop detections clipped at 2+ frame borders (corner/edge partials,
             e.g. a head-only crop that cannot be reliably tracked or recognized).
        """
        if len(detections) > 1:
            keep = [True] * len(detections)
            for i in range(len(detections)):
                if not keep[i]:
                    continue
                ax1, ay1, aw, ah = detections[i][0], detections[i][1], detections[i][2], detections[i][3]
                for j in range(i + 1, len(detections)):
                    if not keep[j]:
                        continue
                    bx1, by1, bw, bh = detections[j][0], detections[j][1], detections[j][2], detections[j][3]
                    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    aa = aw * ah
                    ba = bw * bh
                    small = min(aa, ba)
                    if small > 0 and inter / small >= self.dedup_cover_ratio:
                        if aa < ba:
                            keep[i] = False
                            break
                        keep[j] = False
            detections = [d for d, k in zip(detections, keep) if k]

        out = []
        for d in detections:
            x1, y1, w, h = d[0], d[1], d[2], d[3]
            borders = 0
            if x1 <= self.edge_margin:
                borders += 1
            if y1 <= self.edge_margin:
                borders += 1
            if x1 + w >= frame_w - self.edge_margin:
                borders += 1
            if y1 + h >= frame_h - self.edge_margin:
                borders += 1
            if borders < 2:
                out.append(d)
        return out


def run_detection(
    video_path,
    output_path,
    conf_threshold=0.35,
    imgsz=960,
    device='cuda',
    min_area=900,
    min_height=50,
    min_aspect=1.0,
    max_aspect=4.5,
    min_area_ratio=0.0008,
    dedup_cover_ratio=0.9,
    edge_margin=2,
):
    """
    Run person detection on entire video
    
    Args:
        video_path: Input video path
        output_path: Output JSON path
        conf_threshold: Detection confidence threshold
        imgsz: YOLO input image size
        device: 'cuda' or 'cpu'
        dedup_cover_ratio: Suppress a detection when it is at least this
            fraction covered by a larger detection (removes duplicate boxes)
        edge_margin: Drop detections touching this many frame borders
    
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
        dedup_cover_ratio=dedup_cover_ratio,
        edge_margin=edge_margin,
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
