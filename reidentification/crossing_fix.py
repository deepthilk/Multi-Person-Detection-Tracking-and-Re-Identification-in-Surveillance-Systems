"""
Crossing / occlusion identity correction.
============================================

DeepSORT keeps track ids alive through a merge (IoU -> ~1.0) and re-assigns
them to the two bodies when the merged blob splits. When the two people look
alike (identical uniforms) that re-assignment is effectively a coin-flip: the
id that carried identity X before the crossing can come out following the OTHER
person's body afterwards. Because the face-based wanted filter names a WHOLE
track, the name voted from X's pre-crossing frames then rides the wrong body -
a false box on a person who was never registered.

This module is the tracker-level fix applied as a deterministic post-pass on
the RAW per-camera tracking (it never edits the DeepSORT internals or any
teammate module). It detects every merge / occlusion event between track pairs
and corrects the id-level consequence using face evidence ONLY, so it
generalises to any camera / video:

  * Termination (split): a track was face-confirmed as registered person X
    before the merge, but during / right after the merge its box contains a
    face that is confidently NOT X (a different registered person, or a face
    that matches NO registered person = an unknown third face). The track's
    label jumped bodies. The track is SPLIT at the merge: frames before it
    keep the id (and therefore the name), frames from the merge onward get a
    fresh id so the wanted filter treats them as a new, unnamed person and
    drops them. Handles "wanted person crossed and left the frame; the only
    visible face after is the other person's".

  * Transfer (handoff): same contradiction, but the OTHER merged track's box
    after the merge is face-confirmed to be X. Then X did not leave - the
    tracker just moved her onto the other id. The post-merge segment of that
    track is re-labelled back onto the original id, so the name keeps riding
    X's real body.

  * Swap: both tracks were face-confirmed before the merge (X and Y) and the
    merge-window faces confirm the cross-pairing (A's blob now shows Y, B's
    blob now shows X) - the two people exchanged ids. The post-merge segments
    are exchanged so each track stays one consistent person.

Both actions are purely evidence-driven (faces), camera-agnostic and
idempotent (once a track is split its segments no longer merge the same way,
and unnamed segments have no registered pre-identity to anchor a new action).

This runs in run_integrated_pipeline.py as STEP 1.25, right after tracking and
before the wanted-person filter, so every downstream stage (raw Re-ID identity
links, wanted filter, per-camera Re-ID, cross-camera matching, naming,
rendering) sees the corrected ids.
"""

import logging

import cv2
import numpy as np

from reidentification.face_cue import FaceCueExtractor
from reidentification.face_encoder import confirm_distance as _encoder_confirm_distance
from reidentification.face_name_resolver import (
    _cam_to_video, _source_available, _effective_threshold,
)
from registration.face_db import FaceIdentityDB, face_distance

logger = logging.getLogger(__name__)

# Two boxes that overlap at least this much are "merged".
MERGE_IOU = 0.60
# A merge must last this many consecutive frames to be a real crossing/occlusion
# rather than a momentary box graze.
MIN_RUN = 2
# A face whose best distance to EVERY registered person is above this is an
# unknown third face (the footage's own-face band sits just below the confirm
# cutoff; unknown faces land 0.1+ above it), i.e. proof the box is NOT the
# registered person. Relative to the active encoder's confirm distance.
UNKNOWN_DIST = 0.50
# Margin above the active encoder's confirm distance used for UNKNOWN_DIST when
# the caller did not pass one (keeps 0.50 for dlib, ~0.55 for ArcFace).
UNKNOWN_MARGIN = 0.10
# Frames to keep checking after the merge ends for a contradicting face.
WINDOW_EXTRA = 3
# Max (frame, bbox) samples per segment when extracting faces.
MAX_SAMPLES = 16


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (min(area_a, area_b) + 1e-9)


def _find_merge_events(frames_by_id):
    """Return [(tid_a, tid_b, f_enter, f_exit)] for every track pair that keeps
    IoU >= MERGE_IOU for >= MIN_RUN consecutive frames (ids sorted)."""
    all_frames = sorted({f for fm in frames_by_id.values() for f in fm})
    runs = {}
    events = []
    for f in all_frames:
        present = sorted(t for t, fm in frames_by_id.items() if f in fm)
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                key = (a, b)
                if _iou(frames_by_id[a][f], frames_by_id[b][f]) >= MERGE_IOU:
                    if key not in runs:
                        runs[key] = [f, f]
                    else:
                        runs[key][1] = f
                else:
                    st, en = runs.pop(key, (0, -1))
                    if st > 0 and en - st + 1 >= MIN_RUN:
                        events.append((a, b, st, en))
    for (a, b), (st, en) in runs.items():
        if st > 0 and en - st + 1 >= MIN_RUN:
            events.append((a, b, st, en))
    events.sort(key=lambda e: (e[2], e[0]))
    return events


def _sample(frames, n=MAX_SAMPLES):
    frames = sorted(frames, key=lambda x: x[0])
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def _extract_and_classify(cap, frames, extractor, face_db, persons, threshold):
    """Extract a face per sampled (frame, bbox) and classify each against the
    registered DB. Returns [(frame, best_name, best_distance)]."""
    out = []
    for f, bbox in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, img = cap.read()
        if not ok:
            continue
        enc = extractor.extract(img, bbox)
        if enc is None:
            continue
        best_name, best_dist = None, None
        for name in persons:
            d = face_distance(face_db.encodings_for(name), enc)
            if d is not None and (best_dist is None or d < best_dist):
                best_name, best_dist = name, d
        if best_name is not None:
            out.append((f, best_name, best_dist))
    return out


def _pre_identity(pre_faces, threshold):
    """Registered identity anchored before the merge: the closest confident
    (<= threshold) registered match, or None."""
    if not pre_faces:
        return None
    best = min(pre_faces, key=lambda t: t[2])
    return best[1] if best[2] <= threshold else None


def _contradicts(win_faces, name, threshold, unknown_dist):
    """The merge-window box contains a face that is confidently NOT `name`.

    The window faces are the shared/just-split blob, so they are treated
    conservatively: if ANY face still confirms `name` (a confident match), the
    person is still visibly present and there is NO contradiction (a slightly
    noisy face at 0.4-0.5 that still best-matches `name` is not proof of a
    different body). A contradiction only exists when every extracted face
    fails to confirm `name` AND at least one of them is a confident different
    registered person or an unknown third face."""
    if not win_faces:
        return False
    if any(bn == name and bd <= threshold for _, bn, bd in win_faces):
        return False
    for _, bn, bd in win_faces:
        if bd > unknown_dist:
            return True
        if bd <= threshold and bn != name:
            return True
    return False


def _confirms(win_faces, name, threshold):
    return any(bn == name and bd <= threshold for _, bn, bd in win_faces)


def _relabel_from(boxes, tid, from_frame, new_id):
    """Move every box of `tid` at/after from_frame onto `new_id`."""
    if tid not in boxes:
        return
    moved = [f for f in boxes[tid] if f >= from_frame]
    for f in moved:
        boxes.setdefault(new_id, {})[f] = boxes[tid].pop(f)
    if not boxes[tid]:
        del boxes[tid]


def _swap_post_segments(boxes, a, b, f_exit, next_id):
    """Both people stayed visible and exchanged ids: swap the post-merge
    segments so each id is one consistent person. Returns the new next_id."""
    a_post = {f: boxes[a][f] for f in boxes[a] if f > f_exit}
    b_post = {f: boxes[b][f] for f in boxes[b] if f > f_exit}
    for f in a_post:
        del boxes[a][f]
    for f in b_post:
        del boxes[b][f]
    for f, bbox in b_post.items():
        boxes[a][f] = bbox
    for f, bbox in a_post.items():
        boxes[b][f] = bbox
    return next_id


def _transfer_identity(boxes, carried_tid, lost_tid, f_exit, next_id):
    """`carried_tid`'s pre-merge person actually continued under `lost_tid`'s
    post-merge segment (face-confirmed): move that segment back onto
    `carried_tid`, and give `lost_tid`'s remaining post-merge boxes a fresh id.
    Returns the new next_id."""
    post = [f for f in boxes[lost_tid] if f > f_exit]
    for f in post:
        boxes.setdefault(carried_tid, {})[f] = boxes[lost_tid].pop(f)
    if not boxes[lost_tid]:
        del boxes[lost_tid]
    return next_id


def correct_crossing_identities(tracking_by_cam, cameras, face_db=None,
                                threshold=None,
                                merge_iou=MERGE_IOU, min_run=MIN_RUN,
                                unknown_dist=None,
                                window_extra=WINDOW_EXTRA):
    """
    Fix tracker id swaps caused by crossings / occlusions on the RAW per-camera
    tracking. Camera-agnostic: works on whatever cameras `cameras` describes.

    Args:
        tracking_by_cam: {cam_id: {frame: [{"id": .., "bbox": [x1,y1,x2,y2]}]}}
                         (the pre-filter, full tracking - ids may be int/str).
        cameras:         list of {"camera_id", "source"} (camera_config.CAMERAS).
        face_db:         FaceIdentityDB (default: outputs/registration/face_db.json).

    Returns:
        A corrected copy of tracking_by_cam with the same schema. Untouched
        cameras are returned as-is.
    """
    face_db = face_db or FaceIdentityDB()
    threshold = _effective_threshold(threshold)
    if unknown_dist is None:
        unknown_dist = _encoder_confirm_distance() + UNKNOWN_MARGIN
    persons = face_db.list_persons()
    if not persons:
        logger.info("  Crossing fix: face DB empty - skipping")
        return tracking_by_cam

    cam_video = _cam_to_video(cameras)
    output = {}
    n_splits = n_swaps = n_transfers = 0

    for cam_id, data in tracking_by_cam.items():
        # Normalise to {tid(int): {frame(int): bbox}}.
        boxes = {}
        for k, people in data.items():
            try:
                f = int(k)
            except (TypeError, ValueError):
                continue
            if not isinstance(people, list):
                continue
            for p in people:
                if not isinstance(p, dict):
                    continue
                try:
                    tid = int(p["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                b = p.get("bbox")
                if not b or len(b) < 4:
                    continue
                boxes.setdefault(tid, {})[f] = [int(v) for v in b[:4]]

        events = _find_merge_events(boxes)
        if not events:
            output[cam_id] = data
            continue

        video = cam_video.get(cam_id)
        if not video or not _source_available(video):
            logger.warning(f"  Crossing fix {cam_id}: source unavailable ({video}) - leaving tracking as-is")
            output[cam_id] = data
            continue
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            logger.warning(f"  Crossing fix {cam_id}: could not open {video} - leaving tracking as-is")
            output[cam_id] = data
            continue

        extractor = FaceCueExtractor(upsample_times=1)
        next_id = (max(boxes) + 1) if boxes else 1
        try:
            for a, b, f_enter, f_exit in events:
                pre_a = _sample([(f, bbox) for f, bbox in boxes[a].items() if f < f_enter])
                pre_b = _sample([(f, bbox) for f, bbox in boxes[b].items() if f < f_enter])
                win_a = _sample([(f, bbox) for f, bbox in boxes[a].items()
                                 if f_enter <= f <= f_exit + window_extra])
                win_b = _sample([(f, bbox) for f, bbox in boxes[b].items()
                                 if f_enter <= f <= f_exit + window_extra])

                fa = _extract_and_classify(cap, pre_a, extractor, face_db, persons, threshold)
                fb = _extract_and_classify(cap, pre_b, extractor, face_db, persons, threshold)
                wa = _extract_and_classify(cap, win_a, extractor, face_db, persons, threshold)
                wb = _extract_and_classify(cap, win_b, extractor, face_db, persons, threshold)

                ida = _pre_identity(fa, threshold)
                idb = _pre_identity(fb, threshold)
                if wa or wb:
                    logger.info(
                        f"  Crossing fix {cam_id}: t{a}/t{b} merged f{f_enter}-{f_exit} | "
                        f"pre-a='{ida}' ({len(fa)} faces) pre-b='{idb}' ({len(fb)} faces) | "
                        f"win-a={[(bn, round(bd,2)) for _, bn, bd in wa]} "
                        f"win-b={[(bn, round(bd,2)) for _, bn, bd in wb]}")

                if (ida and idb and ida != idb
                        and _confirms(wa, idb, threshold) and _confirms(wb, ida, threshold)
                        and not _confirms(wa, ida, threshold) and not _confirms(wb, idb, threshold)):
                    next_id = _swap_post_segments(boxes, a, b, f_exit, next_id)
                    n_swaps += 1
                    logger.info(f"  Crossing fix {cam_id}: t{a} ('{ida}') and t{b} ('{idb}') "
                                f"swapped ids at the merge - post-merge segments exchanged")
                    continue

                if ida and _contradicts(wa, ida, threshold, unknown_dist):
                    if _confirms(wb, ida, threshold):
                        next_id = _transfer_identity(boxes, a, b, f_exit, next_id)
                        n_transfers += 1
                        logger.info(f"  Crossing fix {cam_id}: t{a} ('{ida}') actually continued "
                                    f"under t{b} after the merge - name handed off")
                    else:
                        _relabel_from(boxes, a, f_enter, next_id)
                        next_id += 1
                        n_splits += 1
                        logger.info(f"  Crossing fix {cam_id}: t{a} ('{ida}') box now holds a "
                                    f"different face after the merge - track SPLIT at f{f_enter} "
                                    f"(post-merge boxes -> new id)")
                elif idb and _contradicts(wb, idb, threshold, unknown_dist):
                    if _confirms(wa, idb, threshold):
                        next_id = _transfer_identity(boxes, b, a, f_exit, next_id)
                        n_transfers += 1
                        logger.info(f"  Crossing fix {cam_id}: t{b} ('{idb}') actually continued "
                                    f"under t{a} after the merge - name handed off")
                    else:
                        _relabel_from(boxes, b, f_enter, next_id)
                        next_id += 1
                        n_splits += 1
                        logger.info(f"  Crossing fix {cam_id}: t{b} ('{idb}') box now holds a "
                                    f"different face after the merge - track SPLIT at f{f_enter} "
                                    f"(post-merge boxes -> new id)")
        finally:
            cap.release()

        rebuilt = {}
        for tid, frames in boxes.items():
            for f, bbox in frames.items():
                rebuilt.setdefault(f, []).append({"id": tid, "bbox": bbox})
        for f in rebuilt:
            rebuilt[f].sort(key=lambda e: e["id"])
        output[cam_id] = rebuilt

    if n_splits or n_swaps or n_transfers:
        logger.info(f"  Crossing fix: {n_splits} track split(s), {n_swaps} id swap(s), "
                    f"{n_transfers} identity handoff(s) across {len(output)} camera(s)")
    else:
        logger.info("  Crossing fix: no crossing/occlusion identity corrections needed")
    return output
