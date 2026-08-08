"""
Detection diagnostic tool.

Answers the question: "Why does my system only detect 2 of the 4 people?"

Two modes:
  1. Analyze an existing detections.json (default).
  2. Run a LOW-confidence detection pass on a video first, then analyze it.
     YOLO reports a lot more boxes at conf=0.10 than at conf=0.35, so this
     reveals people the current pipeline is silently filtering out.

For every detection it reports:
  - confidence (YOLO score)
  - size (w x h), area
  - aspect ratio (h / w)
  - which pipeline gate would drop it:
      * CONF      score < conf_threshold            (detect_module.py:93)
      * HEIGHT    h < min_height                    (detect_module.py:86)
      * AREA      area < max(min_area, frame_area*ratio)  (detect_module.py:86)
      * ASPECT    aspect < min_aspect or > max_aspect     (detect_module.py:89)

Usage:
  # Analyze an existing file
  python diagnose_detections.py --detections outputs/detections.json

  # Run a fresh low-conf pass, then analyze
  python diagnose_detections.py --video input/video4.mp4

  # Custom thresholds (match config.py / main.py defaults shown here)
  python diagnose_detections.py --video input/video4.mp4 --conf 0.10 \
      --conf-threshold 0.25 --min-height 30 --min-area 600 --min-area-ratio 0.0008
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Low-confidence detection pass (mirrors detect_module.PersonDetector)
# ─────────────────────────────────────────────────────────────────────────────

def run_low_conf_detection(video_path, output_path, conf, imgsz, device):
    """Run YOLOv8 at a low conf threshold, keeping only person boxes but with
    NO filtering (no min area / height / aspect gates)."""
    import torch
    from ultralytics import YOLO

    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'

    logger.info(f"Loading YOLOv8 from models/yolov8s.pt on {device}")
    model = YOLO('models/yolov8s.pt')
    model.to(device)

    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return False

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video {frame_w}x{frame_h}, running raw YOLO at conf={conf} (no filters)")

    frame_id = 0
    all_detections = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        results = model(frame, conf=conf, imgsz=imgsz, device=device)[0]
        dets = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            for box, score, cls in zip(boxes, scores, classes):
                if int(cls) != 0:   # person class only
                    continue
                x1, y1, x2, y2 = map(int, box)
                dets.append([x1, y1, x2 - x1, y2 - y1, float(score)])
        all_detections[frame_id] = dets
        if frame_id % 50 == 0:
            logger.info(f"Frame {frame_id}: {len(dets)} raw detections")

    cap.release()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_detections, f, indent=4)

    max_any = max((len(d) for d in all_detections.values()), default=0)
    logger.info(f"Raw pass done: {frame_id} frames, "
                f"max {max_any} people in one frame")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis
# ─────────────────────────────────────────────────────────────────────────────

def gate_checks(det, conf_threshold, min_height, min_area, min_area_ratio,
                frame_w, frame_h):
    """Return the list of pipeline gates this detection would fail."""
    x, y, w, h, score = det
    area = w * h
    aspect = h / max(w, 1)
    min_area_dynamic = max(min_area, int(frame_w * frame_h * min_area_ratio))

    dropped = []
    if score < conf_threshold:
        dropped.append(f"CONF (score {score:.2f} < {conf_threshold})")
    if h < min_height:
        dropped.append(f"HEIGHT ({h} < {min_height}px)")
    if area < min_area_dynamic:
        dropped.append(f"AREA ({area} < {min_area_dynamic})")
    if aspect < 0.5 or aspect > 4.5:
        dropped.append(f"ASPECT ({aspect:.2f}, must be 0.5-4.5)")
    return dropped


def analyze(detections_path, conf_threshold, min_height, min_area,
            min_area_ratio, top_n):
    with open(detections_path, 'r') as f:
        data = json.load(f)

    # Determine frame size from the largest box seen (estimate only; stored
    # detections don't carry frame dims).
    max_x2, max_y2 = 0, 0
    for dets in data.values():
        for d in dets:
            x, y, w, h, _ = d
            max_x2 = max(max_x2, x + w)
            max_y2 = max(max_y2, y + h)
    frame_w, frame_h = max(max_x2, 1), max(max_y2, 1)

    total_frames = len(data)
    all_dets = [d for dets in data.values() for d in dets]
    counts = {}
    for dets in data.values():
        n = len(dets)
        counts[n] = counts.get(n, 0) + 1

    scores = [d[4] for d in all_dets]
    areas = [d[2] * d[3] for d in all_dets]
    aspects = [d[3] / max(d[2], 1) for d in all_dets]

    per_frame = {int(k): len(v) for k, v in data.items()}
    max_count = max(per_frame.values(), default=0)
    max_frames = sorted(f for f, n in per_frame.items() if n == max_count)

    print()
    print("=" * 78)
    print(f"DETECTION DIAGNOSTIC  —  {detections_path}")
    print("=" * 78)
    print(f"Frames analyzed          : {total_frames}")
    print(f"Total detections (all fr): {len(all_dets)}")
    if total_frames:
        print(f"Avg detections / frame   : {len(all_dets) / total_frames:.2f}")
    print(f"Estimated frame size     : {frame_w}x{frame_h}")
    print()
    print("Per-frame count histogram (number of people detected):")
    for n in sorted(counts):
        frac = counts[n] / total_frames * 100 if total_frames else 0
        print(f"  {n} person(s): {counts[n]:6d} frames  ({frac:5.1f}%)")
    print()
    print(f"MAX people in one frame  : {max_count}")
    print(f"  -> reached in {len(max_frames)} frame(s), e.g. {max_frames[:10]}")
    print()

    # ── Gate simulation ─────────────────────────────────────────────────────
    print("What the current pipeline gates would drop:")
    kept = []
    dropped_by = {}
    below_conf = 0
    for det in all_dets:
        gates = gate_checks(det, conf_threshold, min_height, min_area,
                            min_area_ratio, frame_w, frame_h)
        if not gates:
            kept.append(det)
        else:
            for g in gates:
                tag = g.split('(')[0].strip()
                dropped_by[tag] = dropped_by.get(tag, 0) + 1
        if det[4] < conf_threshold:
            below_conf += 1
    print(f"  KEPT by current gates     : {len(kept)}  "
          f"({(len(kept) / len(all_dets) * 100 if all_dets else 0):.1f}% of raw)")
    for tag, n in sorted(dropped_by.items()):
        print(f"  DROPPED by {tag:<8}: {n}")
    print()
    print(f"Detections below conf {conf_threshold} "
          f"(the 'missing' people candidates): {below_conf}")
    print()

    # ── Score / size distribution ───────────────────────────────────────────
    if scores:
        print("Confidence distribution:")
        for lo, hi in [(0.0, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.50),
                       (0.50, 0.75), (0.75, 1.01)]:
            n = sum(1 for s in scores if lo <= s < hi)
            print(f"  [{lo:.2f}-{hi:.2f}): {n:6d}")
        print(f"  min={min(scores):.2f}  median={sorted(scores)[len(scores)//2]:.2f}  "
              f"max={max(scores):.2f}")
        print()
        print("Aspect ratio distribution (h/w; gate requires 1.0-4.5):")
        for lo, hi in [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.5), (4.5, 99)]:
            n = sum(1 for a in aspects if lo <= a < hi)
            print(f"  [{lo:.1f}-{hi:.1f}): {n:6d}")
        print()

    # ── Top frames detail ───────────────────────────────────────────────────
    print(f"Detail for the {top_n} busiest frames (most people detected):")
    busy = sorted(per_frame.items(), key=lambda kv: -kv[1])[:top_n]
    for fid, n in busy:
        print(f"  Frame {fid}: {n} detection(s)")
        for det in sorted(data[str(fid)], key=lambda d: -d[4]):
            x, y, w, h, score = det
            gates = gate_checks(det, conf_threshold, min_height, min_area,
                                min_area_ratio, frame_w, frame_h)
            status = "KEEP" if not gates else "DROP " + ", ".join(gates)
            print(f"    conf={score:.2f}  box=({x},{y},{w}x{h})  "
                  f"area={w*h}  aspect={h/max(w,1):.2f}  -> {status}")

    # ── Distinct-people estimate ────────────────────────────────────────────
    if all_dets:
        print()
        print("Distinct-people estimate (centroid linkage, dist <= 100px "
              "across consecutive frames):")
        trackers = []   # list of (cx, cy, last_frame, kept_count)
        for fid in sorted(per_frame):
            for det in data[str(fid)]:
                x, y, w, h, _ = det
                cx, cy = x + w / 2, y + h / 2
                best = None
                for t in trackers:
                    if abs(fid - t[2]) <= 5 and t[0] is not None:
                        dx, dy = cx - t[0], cy - t[1]
                        if dx * dx + dy * dy <= 100 * 100:
                            if best is None or (dx * dx + dy * dy) < best[1]:
                                best = (t, dx * dx + dy * dy)
                if best is not None:
                    best[0][0], best[0][1], best[0][2] = cx, cy, fid
                    best[0][3] += 1
                else:
                    trackers.append([cx, cy, fid, 1])
        distinct = sum(1 for t in trackers if t[3] >= 3)
        print(f"  Rough distinct people seen at conf {conf_threshold}: {distinct}")
        print("  (Only tracks that appeared on >= 3 frames count as a person.)")

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description='Diagnose person detection')
    parser.add_argument('--detections', default='outputs/detections.json',
                        help='Existing detections.json to analyze')
    parser.add_argument('--video', default=None,
                        help='Video to run a low-conf detection pass on '
                             '(overwrites --detections with raw output)')
    parser.add_argument('--conf', type=float, default=0.10,
                        help='Low conf for the raw pass (default 0.10)')
    parser.add_argument('--imgsz', type=int, default=960,
                        help='YOLO input size (default 960)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Pipeline conf gate (default 0.25)')
    parser.add_argument('--min-height', type=int, default=30,
                        help='Pipeline min_height gate (default 30)')
    parser.add_argument('--min-area', type=int, default=600,
                        help='Pipeline min_area gate (default 600)')
    parser.add_argument('--min-area-ratio', type=float, default=0.0008,
                        help='Pipeline min_area_ratio (default 0.0008)')
    parser.add_argument('--top', type=int, default=10,
                        help='Busiest frames to print in detail (default 10)')

    args = parser.parse_args()

    if args.video:
        logger.info("=" * 60)
        logger.info("STEP 1: Running RAW low-confidence detection pass")
        logger.info("=" * 60)
        if not run_low_conf_detection(args.video, args.detections, args.conf,
                                      args.imgsz, args.device):
            sys.exit(1)

    if not Path(args.detections).exists():
        logger.error(f"No detections file found at {args.detections}. "
                     "Run with --video first, or point --detections at an "
                     "existing file.")
        sys.exit(1)

    analyze(args.detections, args.conf_threshold, args.min_height,
            args.min_area, args.min_area_ratio, args.top)


if __name__ == "__main__":
    main()
