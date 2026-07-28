"""
Track-level Re-ID: global clustering instead of frame-by-frame decisions.
=============================================================================

WHY THIS EXISTS
----------------
reid_main.py's ReIDEngine makes a fresh identity decision on EVERY SINGLE
FRAME, in real time, blending motion + IOU + body appearance + face into one
score, gated by 5+ interacting thresholds (switch-guard, grace period,
reappearance threshold, continuity lock, face-mismatch veto). Every bug
found debugging this project came from two of those thresholds interacting
badly in some specific edge case (a crossing, a brief occlusion, a person
re-entering frame). That's not bad luck — it's the predictable cost of a
system that has to make an irreversible judgment call every frame from a
single, often-noisy, single-frame observation.

This module replaces that per-frame decision-making with a much simpler,
more robust two-stage design:

  1. Let DeepSORT finish tracking the WHOLE video first (unchanged — Lekha's
     multicam module already does this correctly). This gives a handful of
     track segments per camera (typically 10-30 for a short video with a
     few people), each with a track_id, a start/end frame, and a bbox per
     frame it was seen in.

  2. For each track segment (NOT each frame), extract EVERY body descriptor
     and face descriptor across all its frames, then average them into ONE
     representative body descriptor and pick the single BEST (highest
     confidence) face descriptor for that track. Averaging over dozens of
     frames is inherently far less noisy than trusting any single frame.

  3. Do ONE global comparison across all tracks (from all cameras at once —
     this also replaces the separate cross_camera_match.py step, since
     there's no reason to treat "different camera" specially once matching
     happens at the track level instead of the frame level) and merge
     tracks whose descriptors are confidently similar into one global
     identity. A hard, exact safety rule is applied first: two tracks from
     the SAME camera that were both active during any overlapping frame
     range can NEVER be merged (a single person cannot be two
     simultaneously-visible boxes), which rules out an entire class of
     accidental merges for free.

No online decision, no grace period, no per-frame switch-guard — one clean
offline clustering pass over averaged, far more reliable descriptors.

Not a silver bullet: extreme occlusion where DeepSORT itself never recovers
a track can't be fixed here (that's upstream of any Re-ID logic), and two
genuinely very-similar-looking people with no face detected on either track
could still be merged incorrectly. But it removes the specific, repeated
failure mode this project kept hitting.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from reidentification.reid_main import ReIDEngine, _cosine, _normalize
from reidentification.face_cue import FaceCueExtractor

logger = logging.getLogger(__name__)

# Similarity required to merge two tracks into one identity.
# Face-based comparisons are far more reliable than body appearance
# (especially under identical uniforms), so they get a much lower bar.
MERGE_THRESHOLD_FACE: float = 0.45
MERGE_THRESHOLD_BODY: float = 0.62


class TrackDescriptor:
    """Everything needed to represent one DeepSORT track (one camera, one
    track_id) as a single point for global comparison."""

    def __init__(self, camera_id: str, track_id):
        self.camera_id = camera_id
        self.track_id = track_id
        self.frames: list = []            # frame_ids this track appears in
        self._body_feats: list = []
        self._face_candidates: list = []  # (quality_proxy, face_vector)
        self.avg_body = None
        self.best_face = None

    def add(self, frame_id: int, body_feat, face_feat, face_quality: float = 0.0):
        self.frames.append(frame_id)
        if body_feat is not None:
            self._body_feats.append(body_feat)
        if face_feat is not None:
            self._face_candidates.append((face_quality, face_feat))

    def finalize(self):
        if self._body_feats:
            self.avg_body = _normalize(np.mean(np.stack(self._body_feats), axis=0))
        if self._face_candidates:
            self._face_candidates.sort(key=lambda x: x[0], reverse=True)
            self.best_face = self._face_candidates[0][1]

    @property
    def frame_range(self):
        if not self.frames:
            return (None, None)
        return (min(self.frames), max(self.frames))

    def overlaps(self, other: "TrackDescriptor") -> bool:
        """True only if both tracks are in the SAME camera and were active
        during any shared frame range — the hard veto against merging two
        simultaneously-visible people."""
        if self.camera_id != other.camera_id:
            return False
        a0, a1 = self.frame_range
        b0, b1 = other.frame_range
        if a0 is None or b0 is None:
            return False
        return a0 <= b1 and b0 <= a1


def _track_similarity(t1: TrackDescriptor, t2: TrackDescriptor):
    """Returns (similarity, used_face). used_face=True means the comparison
    used the far-more-reliable face signal, so a lower merge threshold
    applies. Returns (None, False) if neither track has any comparable
    descriptor at all."""
    if t1.best_face is not None and t2.best_face is not None:
        face_sim = FaceCueExtractor.similarity(t1.best_face, t2.best_face)
        if face_sim is not None:
            return face_sim, True
    if t1.avg_body is not None and t2.avg_body is not None:
        return float(_cosine(t1.avg_body, t2.avg_body)), False
    return None, False


def build_track_descriptors(camera_id: str, video_path: str, tracking_json_path: str,
                             engine: ReIDEngine, max_frames=None) -> dict:
    """Runs feature extraction (reusing the existing body+face extractors —
    no reimplementation) over every frame of one camera's video, grouped by
    DeepSORT track_id, and returns {track_id: TrackDescriptor}."""
    with open(tracking_json_path) as f:
        tracking_data = json.load(f)

    tracks: dict = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return {}

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if max_frames and frame_id > max_frames:
            break

        for track in tracking_data.get(str(frame_id), []):
            tid  = track['id']
            bbox = track['bbox']
            body_feat = engine.extract_feature(frame, bbox)
            if body_feat is None:
                continue
            face_feat = engine.face_extractor.extract(frame, bbox)
            quality = 0.0
            if face_feat is not None:
                x1, y1, x2, y2 = bbox
                quality = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            if tid not in tracks:
                tracks[tid] = TrackDescriptor(camera_id, tid)
            tracks[tid].add(frame_id, body_feat, face_feat, quality)

        if frame_id % 50 == 0:
            logger.info(f"  [{camera_id}] track-descriptor extraction: frame {frame_id}")

    cap.release()
    for t in tracks.values():
        t.finalize()

    logger.info(f"✅ [{camera_id}] {len(tracks)} track(s) extracted")
    return tracks


def cluster_tracks(all_tracks: list) -> dict:
    """Union-Find global clustering across ALL tracks (all cameras at once).
    Returns {(camera_id, track_id): global_id}."""
    n = len(all_tracks)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    candidate_pairs = []
    all_comparisons = []   # diagnostic: EVERY non-overlapping pair, not just merges
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = all_tracks[i], all_tracks[j]
            if t1.overlaps(t2):
                continue   # hard veto — can never be the same person
            sim, used_face = _track_similarity(t1, t2)
            if sim is None:
                all_comparisons.append((t1, t2, None, False, None))
                continue
            threshold = MERGE_THRESHOLD_FACE if used_face else MERGE_THRESHOLD_BODY
            all_comparisons.append((t1, t2, sim, used_face, threshold))
            if sim >= threshold:
                candidate_pairs.append((sim, i, j, used_face))

    # Diagnostic: log every comparison so a "should have merged but didn't"
    # pair is visible with its actual number, not just "it didn't happen".
    all_comparisons.sort(key=lambda x: (x[2] if x[2] is not None else -1), reverse=True)
    logger.info("Track-pair comparisons (all, highest similarity first):")
    for t1, t2, sim, used_face, threshold in all_comparisons:
        if sim is None:
            logger.info(f"  [{t1.camera_id}:{t1.track_id}] vs [{t2.camera_id}:{t2.track_id}] "
                        f"-> no comparable descriptor (no body/face on one or both sides)")
        else:
            verdict = "MERGE" if sim >= threshold else "no merge"
            logger.info(f"  [{t1.camera_id}:{t1.track_id}] vs [{t2.camera_id}:{t2.track_id}] "
                        f"-> sim={sim:.3f} ({'face' if used_face else 'body'}, "
                        f"threshold={threshold:.2f}) -> {verdict}")

    # Merge highest-confidence pairs first. Before each merge, re-verify no
    # member of one cluster overlaps with any member of the other — needed
    # because transitive merges (A~B, B~C) could otherwise chain two
    # tracks together that individually never overlap but whose CLUSTERS
    # would contain a same-camera time conflict.
    candidate_pairs.sort(key=lambda x: x[0], reverse=True)
    merge_log = []
    for sim, i, j, used_face in candidate_pairs:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        cluster_i = [k for k in range(n) if find(k) == ri]
        cluster_j = [k for k in range(n) if find(k) == rj]
        conflict = any(all_tracks[a].overlaps(all_tracks[b])
                        for a in cluster_i for b in cluster_j)
        if conflict:
            continue
        union(i, j)
        t1, t2 = all_tracks[i], all_tracks[j]
        merge_log.append(f"  merged [{t1.camera_id}:{t1.track_id}] + "
                          f"[{t2.camera_id}:{t2.track_id}]  sim={sim:.3f} "
                          f"({'face' if used_face else 'body'})")

    if merge_log:
        logger.info("Track merges:\n" + "\n".join(merge_log))

    root_to_gid: dict = {}
    next_gid = 1
    track_to_gid: dict = {}
    for i, t in enumerate(all_tracks):
        r = find(i)
        if r not in root_to_gid:
            root_to_gid[r] = next_gid
            next_gid += 1
        track_to_gid[(t.camera_id, t.track_id)] = root_to_gid[r]
    return track_to_gid


def resolve_names_by_body(all_tracks: list, track_to_gid: dict, registered_persons: dict,
                           match_threshold: float = 0.55) -> dict:
    """Matches each global identity's representative body descriptor
    against Prajna's registration DB (698-dim body embeddings — the
    registration module is body-based, not face-based, so this uses body
    descriptors here for compatibility, matching cross_camera_match.py's
    existing convention). Returns {global_id: {"name":..., "similarity":...}}."""
    if not registered_persons:
        return {}
    gid_to_track = {}
    for t in all_tracks:
        gid = track_to_gid.get((t.camera_id, t.track_id))
        if gid is None or t.avg_body is None:
            continue
        # keep the track with the most frames as the representative for this gid
        if gid not in gid_to_track or len(t.frames) > len(gid_to_track[gid].frames):
            gid_to_track[gid] = t

    names = {}
    for gid, t in gid_to_track.items():
        best_name, best_sim = None, 0.0
        for name, reg_desc in registered_persons.items():
            sim = float(_cosine(t.avg_body, reg_desc))
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_name is not None and best_sim >= match_threshold:
            names[gid] = {"name": best_name, "similarity": round(best_sim, 4)}
    return names


def run_track_level_pipeline(camera_configs: list, per_camera_tracking_paths: dict,
                              device: str = "cuda", max_frames=None,
                              registered_persons: dict = None,
                              output_json_path: str = "outputs/cross_camera/global_identities.json"):
    """Top-level entry point — replaces run_reid_pipeline() + cross-camera
    matching with the track-level approach for every configured camera.
    Output schema matches cross_camera_match.py's existing output, so
    utils.render_global_id_video and everything downstream needs no changes."""
    engine = ReIDEngine(device=device)   # one engine, reused for feature extraction only

    all_tracks: list = []
    per_camera_tracks: dict = {}
    for cfg in camera_configs:
        cam_id = cfg["camera_id"]
        if cam_id not in per_camera_tracking_paths:
            continue
        logger.info(f"— {cam_id} ({cfg['source']}) — extracting track descriptors")
        tracks = build_track_descriptors(
            cam_id, cfg["source"], per_camera_tracking_paths[cam_id], engine, max_frames)
        per_camera_tracks[cam_id] = tracks
        all_tracks.extend(tracks.values())

    if not all_tracks:
        logger.error("❌ No tracks extracted — nothing to cluster")
        return {}

    logger.info(f"Clustering {len(all_tracks)} track(s) across {len(per_camera_tracks)} camera(s)...")
    logger.info("Track summary (frames seen, body descriptor?, face descriptor?):")
    for t in all_tracks:
        f0, f1 = t.frame_range
        logger.info(f"  [{t.camera_id}:{t.track_id}] frames {f0}-{f1} "
                    f"({len(t.frames)} total) | body={'yes' if t.avg_body is not None else 'NO'} "
                    f"| face={'yes' if t.best_face is not None else 'no'}")
    track_to_gid = cluster_tracks(all_tracks)

    name_map = {}
    if registered_persons:
        name_map = resolve_names_by_body(all_tracks, track_to_gid, registered_persons)

    combined: dict = {}
    for cam_id, tracking_path in per_camera_tracking_paths.items():
        with open(tracking_path) as f:
            tracking_data = json.load(f)
        cam_out = {}
        for frame_id_str, dets in tracking_data.items():
            frame_out = []
            for det in dets:
                tid = det['id']
                gid = track_to_gid.get((cam_id, tid))
                entry = dict(det)
                entry['global_id'] = gid
                if gid is not None and gid in name_map:
                    entry['name'] = name_map[gid]['name']
                    entry['name_similarity'] = name_map[gid]['similarity']
                else:
                    entry['name'] = None
                frame_out.append(entry)
            cam_out[frame_id_str] = frame_out
        combined[cam_id] = cam_out

    all_gids = sorted(set(track_to_gid.values()))
    combined["global_identities"] = {
        str(gid): {
            "cameras_seen_on": sorted({cam for (cam, _tid), g in track_to_gid.items() if g == gid}),
            **({"name": name_map[gid]["name"], "similarity": name_map[gid]["similarity"]}
               if gid in name_map else {}),
        }
        for gid in all_gids
    }

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    n_named = sum(1 for v in combined["global_identities"].values() if "name" in v)
    logger.info(f"✅ Track-level clustering complete: {len(all_gids)} global identities "
                f"({n_named} matched to a registered name) -> {output_json_path}")
    return combined
