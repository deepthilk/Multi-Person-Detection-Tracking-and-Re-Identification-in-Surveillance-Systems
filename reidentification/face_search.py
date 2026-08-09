"""
Face-based person search across surveillance videos.
======================================================

This is the RELIABLE search path. Matching is done on 128-dim face encodings
(face_recognition) instead of the fine-tuned body Re-ID descriptor, which is
not discriminative enough on this project's small training set (16 identities)
and causes false positives.

How it works
------------
1. index_video():  every person detection (reusing the existing
   outputs/<video>_detections.json from YOLOv8/DeepSORT) is cropped, a face
   is located in the head zone and encoded. The result is cached as
   outputs/face_search/<video>_faces.json.
2. search_video():  the query person's face encodings (from registration
   photos) are compared to every indexed face. A match requires Euclidean
   distance <= threshold (0.55 default - stricter than face_recognition's
   own 0.6, giving near-zero false positives).
3. report/group:    matched frames are grouped into contiguous appearance
   segments per video, each with a from-to time range.
4. render:          annotated highlight videos + a contact sheet that shows
   the query photo next to the matched crops for visual verification.
"""

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from reidentification.face_cue import FaceCueExtractor
from reidentification.face_encoder import confirm_distance as _encoder_confirm_distance
from registration.face_db import (
    face_distance,
    distance_to_similarity,
)

FACE_INDEX_DIR = Path("outputs/face_search")

# Max frame gap allowed inside one appearance segment (person briefly lost
# by the detector but same continuous appearance).
SEGMENT_MAX_GAP = 5

GREEN = (0, 220, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# contact-sheet layout
BG = (247, 247, 247)
DARK = (28, 28, 28)
QUERY_BORDER = (28, 28, 28)
MATCH_BORDER = (0, 160, 0)
SEG_BAND = (0, 130, 65)
THUMB_W = 148
THUMB_H = 168
LABEL_H = 28
BAND_H = 28
HEADER_H = 56
MARGIN = 20
GAP = 10


def index_video(video_path, detections_path, output_path=None):
    """
    Extract a face encoding for every person detection in a video.

    Args:
        video_path: path to the input video.
        detections_path: path to the YOLOv8 detections JSON
                         ({frame: [[x1,y1,w,h,conf], ...]}).
        output_path: where to cache the index (defaults to
                     outputs/face_search/<stem>_faces.json).

    Returns:
        dict with keys video, fps, width, height, detections (list of dicts
        with frame, bbox, encoding).
    """
    det_path = Path(detections_path)
    if not det_path.exists():
        logger.error(f"Detections file not found: {detections_path}")
        return None

    with open(det_path) as f:
        detections = json.load(f)

    video = Path(video_path).stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    extractor = FaceCueExtractor(upsample_times=1)
    indexed = []
    frame_id = 0
    face_attempts = 0
    face_hits = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        for det in detections.get(str(frame_id), []):
            try:
                x1, y1, w, h, _ = [float(v) for v in det[:5]]
            except (TypeError, ValueError):
                continue
            bbox = [int(x1), int(y1), int(x1 + w), int(y1 + h)]
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if bw < 30 or bh < 80:
                continue
            face_attempts += 1
            face_enc = extractor.extract(frame, bbox)
            if face_enc is None:
                continue
            face_hits += 1
            indexed.append({
                "frame": frame_id,
                "bbox": bbox,
                "encoding": [float(v) for v in face_enc],
            })

    cap.release()
    logger.info(f"  {video}: indexed {face_hits}/{face_attempts} person detections "
                f"with a usable face ({100*face_hits/max(1,face_attempts):.0f}%)")

    index_data = {
        "video": video,
        "fps": fps,
        "width": width,
        "height": height,
        "detections": indexed,
    }

    out_path = Path(output_path) if output_path else FACE_INDEX_DIR / f"{video}_faces.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(index_data, f)
    return index_data


def load_index(index_path):
    with open(index_path) as f:
        return json.load(f)


def get_index(video_path, detections_path, force=False):
    """Return a cached face index for a video, building it if needed."""
    video = Path(video_path).stem
    index_path = FACE_INDEX_DIR / f"{video}_faces.json"
    if index_path.exists() and not force:
        return load_index(index_path)
    return index_video(video_path, detections_path, output_path=index_path)


def search_video(index_data, query_encodings, distance_threshold=None):
    """
    Match a query person's face encodings against an indexed video.

    Returns list of match dicts: {frame, bbox, distance, similarity}.
    Empty list if nobody in the video is the query person.
    """
    if distance_threshold is None:
        distance_threshold = _encoder_confirm_distance()
    if not query_encodings:
        return []
    matches = []
    for det in index_data["detections"]:
        cand = np.asarray(det["encoding"], dtype=np.float32)
        dist = face_distance(query_encodings, cand)
        if dist is not None and dist <= distance_threshold:
            matches.append({
                "frame": det["frame"],
                "bbox": det["bbox"],
                "distance": round(dist, 4),
                "similarity": round(distance_to_similarity(dist), 4),
            })
    matches.sort(key=lambda m: m["frame"])
    # If several detections in the same frame matched, keep the best one
    # (two detections on one frame is usually a duplicate detector box).
    deduped = {}
    for m in matches:
        if m["frame"] not in deduped or m["distance"] < deduped[m["frame"]]["distance"]:
            deduped[m["frame"]] = m
    return [deduped[k] for k in sorted(deduped)]


def group_matches(matches, fps, max_gap=SEGMENT_MAX_GAP):
    """
    Group matched frames into contiguous appearance segments.

    Returns list of segment dicts:
      {first_frame, last_frame, first_time, last_time, duration_seconds,
       n_frames, best_similarity, best_distance, best_frame, best_bbox, frames}
    """
    if not matches:
        return []
    segments = []
    current = [matches[0]]
    for m in matches[1:]:
        if m["frame"] - current[-1]["frame"] <= max_gap:
            current.append(m)
        else:
            segments.append(current)
            current = [m]
    segments.append(current)

    result = []
    for seg in segments:
        best = max(seg, key=lambda m: m["similarity"])
        first = seg[0]["frame"]
        last = seg[-1]["frame"]
        result.append({
            "first_frame": first,
            "last_frame": last,
            "first_time": _format_time(first, fps),
            "last_time": _format_time(last, fps),
            "duration_seconds": round((last - first) / fps, 1),
            "n_frames": len(seg),
            "best_similarity": best["similarity"],
            "best_distance": best["distance"],
            "best_frame": best["frame"],
            "best_bbox": best["bbox"],
            "frames": [m["frame"] for m in seg],
            "frame_bboxes": {m["frame"]: m["bbox"] for m in seg},
        })
    return result


def _format_time(frame, fps):
    total = max(0, int(frame / max(1.0, fps)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _draw_annotation(frame, bbox, name, sim, w):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
    label = f"{name} (face {sim:.2f})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    tx, ty = x1, max(y1 - 10, th + 6)
    cv2.rectangle(frame, (tx, ty - th - 6), (tx + tw + 6, ty + 2), BLACK, -1)
    cv2.putText(frame, label, (tx + 3, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    sub = f"{name} FOUND"
    (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    sx = (w - sw) // 2
    cv2.rectangle(frame, (sx - 10, 10), (sx + sw + 10, 10 + sh + 14), BLACK, -1)
    cv2.putText(frame, sub, (sx, 10 + sh + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 3)


def _new_writer(path, fps, size):
    for codec in ("mp4v", "avc1", "H264"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
    return None


def render_highlight_video(video_path, index_data, query_name, segments, output_path):
    """
    Render a highlight video: green box + name on every matched frame.

    Returns (True, n_annotated, total_frames_written).
    """
    matched = {}
    for seg in segments:
        sim = seg["best_similarity"]
        frame_bboxes = seg.get("frame_bboxes", {})
        for m in seg["frames"]:
            bbox = frame_bboxes.get(m)
            if bbox is None:
                det = next((d for d in index_data["detections"] if d["frame"] == m), None)
                bbox = det["bbox"] if det else None
            if bbox is not None:
                matched[m] = (bbox, sim)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open {video_path}")
        return False, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = _new_writer(output_path, fps, (w, h))
    if out is None:
        logger.error(f"Could not create video writer for {output_path}")
        return False, 0, 0

    frame_id = 0
    annotated = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id in matched:
            bbox, sim = matched[frame_id]
            _draw_annotation(frame, bbox, query_name, sim, w)
            annotated += 1
            out.write(frame)
            written += 1

    cap.release()
    out.release()
    logger.info(f"  Highlight video: {output_path} ({annotated} annotated frame(s))")
    return True, annotated, written


def build_contact_sheet(query_images, index_data, segments, video_path, output_path,
                        name=None, matches=None, max_samples_per_segment=6):
    """
    Build a verification contact sheet for the search results.

    Layout (top to bottom):
      * dark header with the query person's name and source video
      * a QUERY row showing the registered photo(s) used for matching
      * one band per appearance segment, with a grid of up to
        max_samples_per_segment crops sampled across the whole appearance,
        each letterboxed (aspect ratio preserved) and framed in green,
        labelled with frame number, timestamp and face similarity.

    The multi-sample strip lets the user confirm the match across the
    appearance instead of trusting a single best crop.
    """
    if not segments:
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open {video_path} for contact sheet")
        return False

    fps = float(index_data.get("fps") or 30.0)
    sim_by_frame = {m["frame"]: m["similarity"] for m in (matches or [])}

    query_thumbs = [_letterbox(im, THUMB_W, THUMB_H) for im in query_images
                    if im is not None and im.shape[0] >= 4 and im.shape[1] >= 4][:6]

    rows = []
    for idx, seg in enumerate(segments):
        thumbs = _sample_crops(cap, seg, sim_by_frame, fps, max_samples_per_segment)
        if not thumbs:
            continue
        rows.append({
            "band": (f"Appearance {idx + 1} · frames {seg['first_frame']}–{seg['last_frame']} · "
                     f"{seg['first_time']} → {seg['last_time']} · {seg['n_frames']} frame(s) · "
                     f"best similarity {seg['best_similarity']:.3f}"),
            "thumbs": thumbs,
        })
    cap.release()

    if not rows and not query_thumbs:
        return False

    cols = max(1, min(6, max(len(query_thumbs) or 0, *([len(r["thumbs"]) for r in rows] or [0]))))
    sheet_w = MARGIN * 2 + cols * THUMB_W + (cols - 1) * GAP

    heights = [HEADER_H]
    if query_thumbs:
        heights += [BAND_H, THUMB_H + LABEL_H]
    for r in rows:
        heights += [BAND_H] * 1 + [THUMB_H + LABEL_H] * math.ceil(len(r["thumbs"]) / cols)
    sheet_h = sum(heights)

    sheet = np.full((sheet_h, sheet_w, 3), BG, dtype=np.uint8)
    y = 0

    _draw_header(sheet, name, Path(video_path).stem, sheet_w)
    y += HEADER_H

    if query_thumbs:
        _draw_band(sheet, y, sheet_w, f"Registered query photo(s) — {len(query_thumbs)}")
        y += BAND_H
        x = MARGIN
        for i, thumb in enumerate(query_thumbs):
            _put_thumb(sheet, thumb, x, y, QUERY_BORDER)
            _draw_label(sheet, x, y, f"#{i + 1}", "query photo", None)
            x += THUMB_W + GAP
        y += THUMB_H + LABEL_H

    for r in rows:
        _draw_band(sheet, y, sheet_w, r["band"], bg=SEG_BAND)
        y += BAND_H
        chunks = [r["thumbs"][i:i + cols] for i in range(0, len(r["thumbs"]), cols)]
        for chunk in chunks:
            x = MARGIN
            for t in chunk:
                _put_thumb(sheet, t["img"], x, y, MATCH_BORDER)
                _draw_label(sheet, x, y, f"#{t['frame']}", t["time"], t["sim"])
                x += THUMB_W + GAP
            y += THUMB_H + LABEL_H

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.info(f"  Contact sheet: {output_path} ({len(rows)} matched segment(s))")
    return True


def _sample_crops(cap, seg, sim_by_frame, fps, max_samples):
    """Sample up to max_samples crops spread across an appearance segment."""
    frames = sorted(seg.get("frame_bboxes", {}).keys())
    if not frames:
        return []
    picks = np.linspace(0, len(frames) - 1, min(max_samples, len(frames)))
    picks = sorted({int(round(p)) for p in picks})
    crops = []
    for i in picks:
        fr = frames[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        bbox = seg["frame_bboxes"][fr]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        pad = 8
        crop = frame[max(0, y1 - pad):min(frame.shape[0], y2 + pad),
                     max(0, x1 - pad):min(frame.shape[1], x2 + pad)]
        crops.append({
            "img": crop,
            "frame": fr,
            "time": _format_time(fr, fps),
            "sim": sim_by_frame.get(fr, seg["best_similarity"]),
        })
    return crops


def _letterbox(img, cell_w, cell_h, bg=BG):
    """Resize an image into a cell preserving its aspect ratio (pad the rest)."""
    h, w = img.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((cell_h, cell_w, 3), bg, dtype=np.uint8)
    x0 = (cell_w - nw) // 2
    y0 = (cell_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _put_thumb(sheet, img, x, y, border):
    cv2.rectangle(sheet, (x, y), (x + THUMB_W, y + THUMB_H), border, 2)
    inner = _letterbox(img, THUMB_W - 4, THUMB_H - 4)
    sheet[y + 2:y + THUMB_H - 2, x + 2:x + THUMB_W - 2] = inner


def _draw_band(sheet, y, w, text, bg=SEG_BAND, fg=WHITE):
    cv2.rectangle(sheet, (0, y), (w, y + BAND_H), bg, -1)
    cv2.putText(sheet, text, (MARGIN, y + BAND_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 2)


def _draw_label(sheet, x, y, line1, line2, sim):
    cv2.putText(sheet, line1, (x + 2, y + THUMB_H + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, DARK, 1)
    if sim is None:
        cv2.putText(sheet, line2, (x + 2, y + THUMB_H + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1)
    else:
        sim_txt = f"sim {sim:.3f}"
        (tw, _), _ = cv2.getTextSize(sim_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(sheet, sim_txt, (x + THUMB_W - 2 - tw, y + THUMB_H + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 120, 0), 1)


def _draw_header(sheet, name, video, w):
    cv2.rectangle(sheet, (0, 0), (w, HEADER_H), DARK, -1)
    title = f"{name} — face search verification" if name else "Face search verification"
    cv2.putText(sheet, title, (MARGIN, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(sheet, f"video: {video} · green-framed crops are matched detections",
                (MARGIN, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
