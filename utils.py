"""
Utility functions for the multi-person tracking and Re-ID system
"""

import cv2
import json
import numpy as np
from pathlib import Path
import logging
import os
import shutil
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    """Save data to JSON file"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"✅ Saved to {file_path}")


def draw_detections(frame, detections):
    """Draw detection boxes on frame"""
    frame_copy = frame.copy()
    
    for det in detections:
        x, y, w, h, score = det
        x2, y2 = x + w, y + h
        
        cv2.rectangle(frame_copy, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_copy, f"{score:.2f}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame_copy


def draw_tracks(frame, tracks):
    """Draw tracking boxes and IDs on frame"""
    frame_copy = frame.copy()
    
    colors = {}  # Cache colors for consistent ID coloring
    
    for track in tracks:
        track_id = track['id']
        x1, y1, x2, y2 = track['bbox']
        
        # Generate consistent color for each ID
        if track_id not in colors:
            colors[track_id] = (np.random.randint(0, 255),
                               np.random.randint(0, 255),
                               np.random.randint(0, 255))
        
        color = colors[track_id]
        
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(f"ID {track_id}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx, ty = x1, max(y1 - 8, th + 4)
        cv2.rectangle(frame_copy, (tx, ty - th - 4), (tx + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(frame_copy, f"ID {track_id}", (tx + 2, ty - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame_copy


def draw_reid_matches(frame, reid_data):
    """Draw Re-ID matching information on frame"""
    frame_copy = frame.copy()

    # De-duplicate per frame to avoid stacked labels/boxes for the same person.
    deduped = []
    best_by_id = {}
    for person in reid_data:
        tid = person['id']
        cid = person.get('consolidated_id', tid)
        bbox = person['bbox']
        matches = person.get('matches', [])
        score = float(matches[0]['similarity']) if matches else 0.0
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)

        candidate = {
            'tid': tid,
            'cid': cid,
            'bbox': bbox,
            'matches': matches,
            'score': score,
            'area': area,
        }

        prev = best_by_id.get(cid)
        if prev is None or (score, area) > (prev['score'], prev['area']):
            best_by_id[cid] = candidate

    # Remove near-duplicate boxes (50% IoU threshold) and keep the stronger one.
    for candidate in sorted(best_by_id.values(), key=lambda x: (x['score'], x['area']), reverse=True):
        keep = True
        for kept in deduped:
            if compute_iou(candidate['bbox'], kept['bbox']) > 0.50:
                keep = False
                break
        if keep:
            deduped.append(candidate)

    for person in deduped:
        tid = person['tid']
        cid = person['cid']
        x1, y1, x2, y2 = person['bbox']
        matches = person.get('matches', [])
        
        color = (0, 255, 255)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
        
        label = f"ID {cid}"
        if matches:
            best_match = matches[0]
            label += f" ({best_match['similarity']:.2f})"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx, ty = x1, max(y1 - 8, th + 4)
        cv2.rectangle(frame_copy, (tx, ty - th - 4), (tx + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(frame_copy, label, (tx + 2, ty - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame_copy


def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou


def visualize_results(video_path, detections_json, tracking_json, 
                     reid_json=None, output_video_path=None, 
                     mode='tracking'):
    """
    Visualize results by drawing on video frames
    
    Args:
        video_path: Input video path
        detections_json: Path to detections JSON
        tracking_json: Path to tracking JSON
        reid_json: Path to Re-ID JSON (optional)
        output_video_path: Save visualized video (optional)
        mode: 'detection', 'tracking', or 'reid'
    """
    cap = cv2.VideoCapture(video_path)
    
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    detections = load_json(detections_json) if detections_json else {}
    tracks = load_json(tracking_json) if tracking_json else {}
    reid_res = load_json(reid_json) if reid_json else {}
    # Handle new format: {"frames": {...}, "consolidated_features": ...}
    if "frames" in reid_res:
        reid_res = reid_res["frames"]
    
    frame_id = 0
    
    logger.info(f"Visualizing video ({mode} mode)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        if mode == 'detection':
            frame_dets = detections.get(str(frame_id), [])
            frame = draw_detections(frame, frame_dets)
        
        elif mode == 'tracking':
            frame_tracks = tracks.get(str(frame_id), [])
            frame = draw_tracks(frame, frame_tracks)
        
        elif mode == 'reid':
            frame_reid = reid_res.get(str(frame_id), [])
            frame = draw_reid_matches(frame, frame_reid)
        
        cv2.imshow(f"{mode.upper()}", frame)
        
        if output_video_path:
            out.write(frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
    
    cap.release()
    if output_video_path:
        out.release()
    cv2.destroyAllWindows()
    
    logger.info("✅ Visualization complete")


def render_reid_video(video_path, reid_json, output_video_path):
    """Render a Re-ID overlay video without opening a display window."""
    reid_res = load_json(reid_json) if reid_json else {}
    if "frames" in reid_res:
        reid_res = reid_res["frames"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    final_path = _browser_playable_path(output_video_path)
    out, final_path = _open_video_writer(final_path, fps, (width, height))
    if out is None:
        cap.release()
        logger.error("Failed to initialize VideoWriter for browser playback")
        return None

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame_reid = reid_res.get(str(frame_id), [])
        frame = draw_reid_matches(frame, frame_reid)
        out.write(frame)

    cap.release()
    out.release()
    if Path(final_path).suffix.lower() == ".mp4":
        _reencode_for_browser(final_path)
    logger.info(f"✅ Rendered Re-ID video: {final_path}")
    return final_path


def _find_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    base_dir = os.environ.get("LOCALAPPDATA")
    if not base_dir:
        return None

    winget_root = Path(base_dir) / "Microsoft" / "WinGet" / "Packages"
    if not winget_root.exists():
        return None

    candidates = winget_root.glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe")
    for candidate in candidates:
        return str(candidate)

    return None


def _browser_playable_path(output_video_path):
    """Choose the file format a web browser will actually play.

    When ffmpeg is unavailable, OpenCV can only write mp4v (MPEG-4 Part 2),
    which Chrome/Firefox/Edge all refuse to play. In that case return a .webm
    sibling path so the renderer writes a browser-native VP8 file instead.
    """
    if _find_ffmpeg():
        return str(output_video_path)
    return str(Path(output_video_path).with_suffix(".webm"))


def _open_video_writer(output_video_path, fps, size):
    """Open a VideoWriter that produces a browser-playable file.

    - .webm target: VP8, which every browser plays natively (OpenCV can encode
      VP8 without any external encoder).
    - .mp4 target: H.264 (avc1) then mp4v; the caller re-encodes the file to
      H.264 with ffmpeg afterwards, so the intermediate encode quality does not
      matter.

    Returns (writer, final_path) or (None, None).
    """
    width, height = size
    if Path(output_video_path).suffix.lower() == ".webm":
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"VP80"), float(fps), (width, height))
        if writer.isOpened():
            return writer, output_video_path
        logger.warning("VP8 webm writer failed - falling back to mp4")
        output_video_path = str(Path(output_video_path).with_suffix(".mp4"))
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*codec), float(fps), (width, height))
        if writer.isOpened():
            return writer, output_video_path
    return None, None


def _reencode_for_browser(output_video_path):
    """Re-encode with ffmpeg to improve browser compatibility if available."""
    ffmpeg_path = _find_ffmpeg()
    if not ffmpeg_path:
        logger.warning("ffmpeg not found; output may not play in browser")
        return

    temp_path = str(Path(output_video_path).with_suffix(".h264.mp4"))
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(output_video_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        temp_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        Path(temp_path).replace(output_video_path)
        logger.info("Re-encoded output with ffmpeg for browser playback")
    except Exception as exc:
        logger.warning(f"ffmpeg re-encode failed: {exc}")


def generate_summary_report(detections_json, tracking_json, reid_json=None):
    """Generate a summary report of the results"""
    
    detections = load_json(detections_json)
    tracks = load_json(tracking_json)
    reid_res = load_json(reid_json) if reid_json else {}
    if "frames" in reid_res:
        reid_res = reid_res["frames"]
    
    # Count statistics
    total_frames = len(detections)
    total_detections = sum(len(dets) for dets in detections.values())
    unique_ids = set()
    
    for frame_tracks in tracks.values():
        for track in frame_tracks:
            unique_ids.add(track['id'])
    
    report = {
        'total_frames': total_frames,
        'total_detections': total_detections,
        'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
        'unique_tracked_ids': len(unique_ids),
        'has_reid_results': len(reid_res) > 0
    }
    
    return report


def draw_global_matches(frame, cam_frame_people):
    """Draw boxes ONLY for wanted (registered) people - i.e. entries that have
    a real `name`. Anyone without a name (not registered) is not drawn at all,
    matching the wanted-person gating of the pipeline."""
    frame_copy = frame.copy()
    seen_ids = set()
    for person in cam_frame_people:
        if not person.get('name'):
            continue  # not a registered/wanted person -> no box, no label
        gid = person.get('global_id')
        if gid is None or gid in seen_ids:
            continue
        seen_ids.add(gid)
        x1, y1, x2, y2 = person['bbox']
        color = (0, 200, 0)
        sim = person.get('name_similarity')
        label = f"{person['name']} ({sim:.2f})" if sim is not None else person['name']
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx, ty = x1, max(y1 - 8, th + 4)
        cv2.rectangle(frame_copy, (tx, ty - th - 4), (tx + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(frame_copy, label, (tx + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame_copy


def render_global_id_video(video_path, combined_json_path, camera_id, output_video_path):
    """Render one camera's video with GLOBAL identity boxes (cross-camera
    consistent IDs / names), reading reidentification/cross_camera_match.py's
    combined output — e.g. outputs/cross_camera/global_identities.json."""
    combined = load_json(combined_json_path)
    cam_results = combined.get(camera_id, {})
    total_detections = sum(len(v) for v in cam_results.values())
    no_detections = total_detections == 0
    if no_detections:
        logger.warning(f"{camera_id}: no persons detected (no registered/wanted faces in this video)")

    def _overlay_no_person(frame):
        """Full-frame 'Person not detected' overlay when the camera saw nobody."""
        if not no_detections:
            return frame
        scale = max(0.9, frame.shape[0] / 600.0)
        text = "Person not detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        cx = (frame.shape[1] - tw) // 2
        cy = frame.shape[0] // 2
        pad = 12
        cv2.rectangle(
            frame,
            (cx - pad, cy - th - pad),
            (cx + tw + pad, cy + baseline + pad),
            (0, 0, 0), -1,
        )
        cv2.putText(frame, text, (cx, cy), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)
        return frame

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or not np.isfinite(fps) or fps <= 0:
        fps = 30  # image-sequence backends often report fps=-1 or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Image-sequence backends can also report wrong dimensions; trust the
    # actual first frame instead (this also advances the reader to frame 2).
    ok0, first = cap.read()
    if ok0 and first is not None:
        height, width = first.shape[:2]

    final_path = _browser_playable_path(output_video_path)
    out, final_path = _open_video_writer(final_path, fps, (width, height))
    if out is None:
        cap.release()
        logger.error("Failed to initialize VideoWriter for browser playback")
        return None

    frame_id = 0
    if ok0 and first is not None:
        frame = first
        frame_id = 1
        frame_people = cam_results.get(str(frame_id), [])
        frame = draw_global_matches(frame, frame_people)
        frame = _overlay_no_person(frame)
        out.write(frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        frame_people = cam_results.get(str(frame_id), [])
        frame = draw_global_matches(frame, frame_people)
        frame = _overlay_no_person(frame)
        out.write(frame)

    cap.release()
    out.release()
    if Path(final_path).suffix.lower() == ".mp4":
        _reencode_for_browser(final_path)
    logger.info(f"✅ Global-ID video saved -> {final_path}")
    return final_path