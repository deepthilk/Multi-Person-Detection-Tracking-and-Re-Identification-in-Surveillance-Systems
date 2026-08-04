"""
Person Tracking Module (DeepSORT)
Tracks detected persons across frames.

CHANGE FROM ORIGINAL:
  max_age:            30  →  5
    Reason: With max_age=30, DeepSORT kept predicting a departed person's
    position for 30 frames after their last real detection. On a 848x480
    frame the Kalman filter extrapolated y-coordinates to 587-677 (entirely
    below the 480px frame boundary), producing bboxes like
    [454, 587, 477, 677] that crash the Re-ID feature extractor.
    Setting max_age=5 expires ghost tracks quickly so they never drift
    far outside the frame boundaries.

  n_init:             3   →  2
    Reason: Faster track confirmation means fewer missed detections at the
    start of a person's appearance. With n_init=2, a track is confirmed
    after 2 consecutive detections instead of 3, reducing the chance that
    a person's first frames are lost to the Re-ID gallery.

  max_cosine_distance: 0.2 →  0.3
    Reason: Slightly looser appearance matching inside DeepSORT lets it
    maintain tracks through brief partial occlusions without dropping and
    re-assigning a new track ID, which would appear to the Re-ID system as
    a new person.
"""

import cv2
import json
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PersonTracker:
    """DeepSORT based multi-object tracker."""

    def __init__(
        self,
        max_age: int            = 5,    # CHANGED from 30 — prevents off-screen ghost bboxes
        n_init: int             = 2,    # CHANGED from 3  — faster track confirmation
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,  # CHANGED from 0.2 — better occlusion handling
    ):
        self.tracker = DeepSort(
            max_age             = max_age,
            n_init              = n_init,
            max_iou_distance    = max_iou_distance,
            max_cosine_distance = max_cosine_distance,
        )
        logger.info(
            f"✅ Tracker initialised  max_age={max_age}  n_init={n_init}  "
            f"max_cosine_distance={max_cosine_distance}"
        )

    def update(self, frame, detections: list) -> list:
        """
        Update tracker with new detections.

        Args:
            frame:      OpenCV BGR frame (used internally by DeepSORT for appearance)
            detections: List of [x, y, w, h, score] detections

        Returns:
            List of confirmed tracks: [{'id': int, 'bbox': [x1,y1,x2,y2]}, ...]
        """
        ds_detections = [([x, y, w, h], score, 'person') for x, y, w, h, score in detections]
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        frame_h, frame_w = frame.shape[:2]
        confirmed = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tl, tt, tr, tb = track.to_ltrb()
            # Clamp to frame boundaries before storing — defensive measure
            # so downstream code never receives off-screen coordinates.
            x1 = max(0, int(tl)); y1 = max(0, int(tt))
            x2 = min(frame_w, int(tr)); y2 = min(frame_h, int(tb))
            # Only keep tracks with a valid visible region (at least 8x8 px)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            confirmed.append({'id': track.track_id, 'bbox': [x1, y1, x2, y2]})

        return confirmed


def run_tracking(video_path: str, detections_path: str, output_path: str) -> dict:
    """
    Run person tracking on video using pre-computed detections.

    Args:
        video_path:      Input video path
        detections_path: Path to detections JSON
        output_path:     Output JSON path

    Returns:
        Dictionary of frame_id -> list of track dicts
    """
    logger.info(f"Loading detections from {detections_path}")
    with open(detections_path, 'r') as f:
        all_detections = json.load(f)

    tracker = PersonTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return {}

    frame_id         = 0
    tracking_results = {}

    logger.info("Processing video with tracking...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id   += 1
        detections  = all_detections.get(str(frame_id), [])
        tracks      = tracker.update(frame, detections)
        tracking_results[frame_id] = tracks

        if frame_id % 50 == 0:
            logger.info(f"Frame {frame_id}: {len(tracks)} active tracks")

    cap.release()

    # Merge nested/fragment tracks: with a two-tier (weak + strong) detection
    # gate, the same physical person can briefly be tracked twice — once from a
    # weak head/shoulders-only box at the frame edge and once from the full-body
    # box that appears a couple of frames later. The fragment's bbox is almost
    # entirely contained inside the full-body bbox over the same frames, so it
    # looks like a NEW person to the Re-ID stage and produces a phantom identity
    # (e.g. "Alice" that is really Prajna). Merge the contained track into the
    # containing one so one person never becomes two.
    tracking_results = _merge_nested_tracks(tracking_results)

    unique_ids = {t['id'] for tracks in tracking_results.values() for t in tracks}
    logger.info(f"✅ Tracking complete: {frame_id} frames, {len(unique_ids)} unique person IDs")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(tracking_results, f, indent=4)

    return tracking_results


def _box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _contained_ratio(inner, outer):
    """Fraction of inner's area that lies inside outer (0.0-1.0)."""
    inner_area = _box_area(inner)
    if inner_area <= 0:
        return 0.0
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return ((ix2 - ix1) * (iy2 - iy1)) / inner_area


def _merge_nested_tracks(tracking_results: dict) -> dict:
    """Drop fragment tracks whose bbox is almost fully contained inside another
    track's bbox on the same frames (nested duplicate). Returns a new dict.

    Only a SMALL, SHORT-LIVED track counts as a fragment. A real person
    crossing behind another has a similarly-sized box (area ratio ~0.4-1.0)
    and an independent life before/after; a duplicate created by the weak-tier
    head/shoulders box is tiny (area ratio ~0.1) and dies within a few frames.
    """
    # Total frames per track (fragments are transient).
    total_frames: dict = {}
    for tracks in tracking_results.values():
        for t in tracks:
            total_frames[t['id']] = total_frames.get(t['id'], 0) + 1

    # Count, for every ordered pair (contained, container), how many shared
    # frames satisfy: inner bbox >= 80% contained in outer AND inner area is
    # at most 35% of outer's area.
    votes: dict = {}
    for frame_id, tracks in tracking_results.items():
        if len(tracks) < 2:
            continue
        boxes = {t['id']: t['bbox'] for t in tracks}
        for a_id, a_box in boxes.items():
            a_area = _box_area(a_box)
            if a_area <= 0:
                continue
            for b_id, b_box in boxes.items():
                if a_id == b_id:
                    continue
                b_area = _box_area(b_box)
                if b_area <= 0:
                    continue
                if a_area > 0.35 * b_area:
                    continue  # similarly-sized box — a real person, not a fragment
                if _contained_ratio(a_box, b_box) >= 0.8:
                    key = (a_id, b_id)
                    votes[key] = votes.get(key, 0) + 1

    # A fragment needs to be nested on several frames (not a one-off overlap)
    # AND be short-lived overall (<= 12 frames) before we trust it.
    drop_ids = set()
    for (a_id, b_id), n in votes.items():
        if n >= 3 and total_frames.get(a_id, 0) <= 12:
            drop_ids.add(a_id)

    if not drop_ids:
        return tracking_results

    logger.info(f"Merging {len(drop_ids)} nested fragment track(s): "
                f"{sorted(drop_ids)} contained inside larger tracks — removing duplicates")

    merged = {}
    for frame_id, tracks in tracking_results.items():
        kept = [t for t in tracks if t['id'] not in drop_ids]
        if kept:
            merged[frame_id] = kept
    return merged


if __name__ == "__main__":
    import sys
    video_path      = sys.argv[1] if len(sys.argv) > 1 else "input/video4.mp4"
    detections_path = "outputs/detections.json"
    output_path     = "outputs/tracking.json"
    run_tracking(video_path, detections_path, output_path)