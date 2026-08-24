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
        
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_copy, f"ID {track_id}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame_copy


def fill_track_gaps(reid_res, max_gap=15):
    """Return a copy of reid_res with every frame from the first to the last
    present, and short per-track gaps filled by linear box interpolation.

    The reid step samples frames (stride > 1), so the output JSON only carries
    boxes on every Nth frame. Rendering those directly makes boxes appear and
    disappear ("blink") between samples. Filling each track's gaps with
    interpolated boxes keeps a box continuously visible that glides toward the
    person's next position instead of vanishing.

    Non-numeric metadata keys (__tracks__, __verify_faces__) are preserved.
    """
    if not isinstance(reid_res, dict) or not reid_res:
        return reid_res

    numeric = {
        int(f): v
        for f, v in reid_res.items()
        if (isinstance(f, int) or (isinstance(f, str) and f.lstrip('-').isdigit()))
    }
    if not numeric:
        return reid_res

    lo, hi = min(numeric), max(numeric)
    filled = {str(f): [] for f in range(lo, hi + 1)}

    timeline = {}
    for fid, people in numeric.items():
        if not isinstance(people, list):
            continue
        for p in people:
            if not isinstance(p, dict):
                continue
            tid = p.get('consolidated_id')
            if tid is None or tid == -1:
                continue
            bbox = p.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            timeline.setdefault(tid, {})[fid] = tuple(float(v) for v in bbox)

    for tid, tl in timeline.items():
        fids = sorted(tl)
        for i, fid in enumerate(fids):
            filled[str(fid)].append({"consolidated_id": tid, "bbox": list(tl[fid])})
            if i + 1 >= len(fids):
                continue
            nxt = fids[i + 1]
            gap = nxt - fid - 1
            if gap <= 0 or gap > max_gap:
                continue
            b0 = tl[fid]
            b1 = tl[nxt]
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                ib = [a + (b - a) * t for a, b in zip(b0, b1)]
                filled[str(fid + k)].append({"consolidated_id": tid, "bbox": ib})

    for key, value in reid_res.items():
        if not (isinstance(key, int) or (isinstance(key, str) and key.lstrip('-').isdigit())):
            filled[key] = value

    return filled


def draw_reid_matches(frame, reid_data, name_map=None, gid_map=None, tentative_map=None):
    """Draw Re-ID matching information on frame.

    Style: solid BLACK box, BLACK label placard. Text is RED for a resolved
    (known) person and GREEN for an unknown person. An unknown track with a
    near-threshold database candidate shows a green "may be <name> (x%)"
    hint instead of a bare ID.
    """
    frame_copy = frame.copy()

    # De-duplicate per frame to avoid stacked labels/boxes for the same person.
    deduped = []
    best_by_id = {}
    for person in reid_data:
        person_id = person.get('consolidated_id') or person.get('id')
        if person_id is None:
            continue
        bbox = person['bbox']
        matches = person.get('matches', [])
        score = float(matches[0]['similarity']) if matches else 0.0
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)

        candidate = {
            'person_id': person_id,
            'bbox': bbox,
            'matches': matches,
            'score': score,
            'area': area,
        }

        prev = best_by_id.get(person_id)
        if prev is None or (score, area) > (prev['score'], prev['area']):
            best_by_id[person_id] = candidate

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
        person_id = person['person_id']
        x1, y1, x2, y2 = [int(v) for v in person['bbox']]

        # Black box: RED text = known person, GREEN text = unknown. A thin
        # white outline keeps the black box visible on dark clothing/scenes.
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 0), 3)

        name = name_map.get(str(person_id)) if name_map is not None else None
        gid = gid_map.get(str(person_id)) if gid_map is not None else None
        tent = tentative_map.get(str(person_id)) if tentative_map is not None else None
        if name and gid is not None:
            label = f"{name} · GID {gid}"
        elif name:
            label = name
        elif tent and tent.get("name"):
            pct = round(float(tent.get("similarity") or 0) * 100)
            label = f"may be {tent['name']} ({pct}%)"
        elif gid is not None:
            label = f"Global ID {gid}"
        else:
            label = f"ID {person_id}"
        color = (0, 0, 255) if name else (0, 255, 0)  # red / green

        # BLACK label placard with colored text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lx1, ly1 = x1, max(0, y1 - th - 10)
        lx2, ly2 = x1 + tw + 8, y1 - 4
        cv2.rectangle(frame_copy, (lx1, ly1), (lx2, ly2), (0, 0, 0), -1)
        cv2.rectangle(frame_copy, (lx1, ly1), (lx2, ly2), (255, 255, 255), 1)
        cv2.putText(frame_copy, label, (lx1 + 4, ly1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
    # Fill per-track gaps so boxes stay continuously visible (no blinking on
    # the stride-sampled reid frames), moving smoothly toward the next sample.
    reid_res = fill_track_gaps(reid_res)

    # Resolved-name map (final_id -> name), populated by the web pipeline's
    # "__tracks__" section and refreshed by cross-camera / manual corrections
    # before re-render. Falls back to no names (old ID-style labels).
    name_map = {}
    gid_map = {}
    tentative_map = {}
    tracks = reid_res.get("__tracks__") if isinstance(reid_res, dict) else None
    if isinstance(tracks, dict):
        name_map = {
            tid: t.get("name")
            for tid, t in tracks.items()
            if t.get("name")
        }
        gid_map = {
            tid: t["global_id"]
            for tid, t in tracks.items()
            if t.get("global_id") is not None
        }
        tentative_map = {
            tid: {"name": t.get("tentative_name"), "similarity": t.get("tentative_similarity")}
            for tid, t in tracks.items()
            if t.get("tentative_name") and not t.get("name")
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    for codec in ("mp4v", "avc1", "H264"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if writer.isOpened():
            out = writer
            logger.info(f"Using video codec: {codec}")
            break

    if out is None:
        cap.release()
        logger.error("Failed to initialize VideoWriter with available codecs")
        return False

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame_reid = reid_res.get(str(frame_id), [])
        frame = draw_reid_matches(frame, frame_reid, name_map, gid_map, tentative_map)
        out.write(frame)

    cap.release()
    out.release()
    _reencode_for_browser(output_video_path)
    logger.info(f"✅ Rendered Re-ID video: {output_video_path}")
    return True


def extract_track_thumbnail(video_path, reid_json_path, track_id, output_path, max_size=96):
    """Save a representative crop of a track as a JPEG thumbnail, for showing
    'who is this person' in the results UI. Picks the frame where the track's
    bounding box is LARGEST (best/clearest view), crops it with a little
    padding and downscales to fit max_size. Returns True on success."""
    reid_res = load_json(reid_json_path) if reid_json_path else {}
    best = None  # (area, frame_id, bbox)
    for fid_str, frame_people in reid_res.items():
        if not isinstance(frame_people, list):
            continue
        try:
            fid = int(fid_str)
        except (TypeError, ValueError):
            continue
        for p in frame_people:
            if not isinstance(p, dict) or p.get('consolidated_id') != track_id:
                continue
            bbox = p.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if best is None or area > best[0]:
                best = (area, fid, (x1, y1, x2, y2))
    if best is None:
        return False

    _, fid, (x1, y1, x2, y2) = best
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fid - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False

    h, w = frame.shape[:2]
    pad = max(8, int(min(x2 - x1, y2 - y1) * 0.12))
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return False

    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    scale = min(1.0, max_size / max(ch, cw))
    if scale < 1.0:
        crop = cv2.resize(
            crop, (max(1, int(cw * scale)), max(1, int(ch * scale))),
            interpolation=cv2.INTER_AREA,
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


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


if __name__ == "__main__":
    print("Utils module for multi-person tracking and Re-ID system")
def draw_global_matches(frame, cam_frame_people):
    frame_copy = frame.copy()
    seen_ids = set()
    for person in cam_frame_people:
        gid = person.get('global_id')
        if gid is None or gid in seen_ids:
            continue  # only skip exact duplicate detections of the SAME id
        seen_ids.add(gid)
        x1, y1, x2, y2 = person['bbox']
        if person.get('name'):
            color = (0, 200, 0)
            label = f"{person['name']} ({person['name_similarity']:.2f})"
        else:
            color = (0, 255, 255)
            label = f"Global ID {gid}"
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_copy, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame_copy


def render_global_id_video(video_path, combined_json_path, camera_id, output_video_path):
    """Render one camera's video with GLOBAL identity boxes (cross-camera
    consistent IDs / names), reading reidentification/cross_camera_match.py's
    combined output — e.g. outputs/cross_camera/global_identities.json."""
    combined = load_json(combined_json_path)
    cam_results = combined.get(camera_id, {})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    for codec in ("mp4v", "avc1", "H264"):
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            out = writer
            break
    if out is None:
        cap.release()
        logger.error("Failed to initialize VideoWriter with available codecs")
        return False

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        frame_people = cam_results.get(str(frame_id), [])
        frame = draw_global_matches(frame, frame_people)
        out.write(frame)

    cap.release()
    out.release()
    _reencode_for_browser(output_video_path)
    logger.info(f"✅ Global-ID video saved -> {output_video_path}")
    return True

