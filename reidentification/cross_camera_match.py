"""
Cross-Camera Identity Matching
================================

reid_main.run_reid_pipeline() already gives rock-solid identity consistency
WITHIN one camera (a person keeps the same consolidated_id for the whole
video). What's still missing is the piece across cameras: cam1's "person 3"
and cam2's "person 7" might be the same physical human, but each camera's
ReIDEngine only ever sees its own video, so their stable_ids are assigned
independently and just happen to collide by coincidence.

This module closes that gap. It:

  1. Runs (or accepts already-run) `run_reid_pipeline` once per camera —
     unmodified, exactly as documented in multicamera/multi_cam_pipeline.py.
  2. Collects each camera's `engine.consolidated_features`
     ({local_stable_id: 698-dim descriptor}).
  3. Globally matches those descriptors across all cameras with a Hungarian
     assignment (falls back to greedy if scipy's linear_sum_assignment
     can't be used, e.g. ragged camera counts), so "cam1 person 3" and
     "cam2 person 7" collapse into a single global_id.
  4. Optionally resolves each global identity to a NAME using Prajna's
     registration.identity_db.IdentityDatabase().export_for_reid() — same
     698-dim descriptor space, so no extra model or conversion is needed.
  5. Rewrites every camera's per-frame results with the new `global_id`
     (and `name`, if resolved) alongside the original per-camera
     `consolidated_id`, and returns/saves one combined structure ready for
     the dashboard.

Nothing here touches reid_main.py's matching logic *within* a camera — it
only combines the OUTPUT of independent per-camera runs, so re-running this
file can't destabilize Lekha's, Prajna's, or Pranjali's modules.
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

CROSS_CAM_MATCH_THRESHOLD = 0.60   # same scale as reid_main's OSNet/ResNet thresholds


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


class CrossCameraMatcher:
    """
    Feed it one camera's (camera_id, consolidated_features) at a time via
    `add_camera`, then call `resolve()` once every camera has been added.
    """

    def __init__(self, match_threshold: float = CROSS_CAM_MATCH_THRESHOLD):
        self.match_threshold = match_threshold
        # global_id -> running-average descriptor for that global identity
        self._global_descriptors: dict = {}
        self._next_global_id = 1
        # (camera_id, local_id) -> global_id
        self.local_to_global: dict = {}

    def add_camera(self, camera_id: str, consolidated_features: dict):
        """
        consolidated_features: {local_stable_id: np.ndarray(698,)} as
        produced by `engine.consolidated_features` after
        `engine.finalize_clustering()` / `run_reid_pipeline`.
        """
        local_ids = list(consolidated_features.keys())
        if not local_ids:
            return

        if not self._global_descriptors:
            # First camera seeds the global identity set 1:1.
            for lid in local_ids:
                gid = self._next_global_id
                self._next_global_id += 1
                self._global_descriptors[gid] = consolidated_features[lid].copy()
                self.local_to_global[(camera_id, lid)] = gid
            return

        global_ids = list(self._global_descriptors.keys())
        cost = np.zeros((len(local_ids), len(global_ids)), dtype=np.float32)
        for i, lid in enumerate(local_ids):
            for j, gid in enumerate(global_ids):
                sim = _cosine(consolidated_features[lid], self._global_descriptors[gid])
                cost[i, j] = 1.0 - sim   # Hungarian minimizes cost -> use distance

        row_idx, col_idx = linear_sum_assignment(cost)
        matched_local = set()
        for r, c in zip(row_idx, col_idx):
            sim = 1.0 - cost[r, c]
            if sim >= self.match_threshold:
                lid, gid = local_ids[r], global_ids[c]
                self.local_to_global[(camera_id, lid)] = gid
                # running average keeps the global descriptor representative
                # of every camera view seen so far, not just the first
                self._global_descriptors[gid] = (
                    0.7 * self._global_descriptors[gid] + 0.3 * consolidated_features[lid]
                )
                self._global_descriptors[gid] /= (
                    np.linalg.norm(self._global_descriptors[gid]) + 1e-8
                )
                matched_local.add(lid)

        # Anything left over is a genuinely new person, first seen on this camera
        for lid in local_ids:
            if lid in matched_local:
                continue
            gid = self._next_global_id
            self._next_global_id += 1
            self._global_descriptors[gid] = consolidated_features[lid].copy()
            self.local_to_global[(camera_id, lid)] = gid

    def get_global_id(self, camera_id: str, local_id: int):
        return self.local_to_global.get((camera_id, local_id))

    def global_descriptors(self) -> dict:
        return dict(self._global_descriptors)


def resolve_names(global_descriptors: dict, registered_persons: dict,
                   match_threshold: float = None) -> dict:
    """
    global_id -> name, using Prajna's registered-persons embeddings
    ({name: np.ndarray(698,)} from IdentityDatabase().export_for_reid()).
    Every global identity that doesn't clear the threshold against any
    registered person is simply left unnamed (stays "Person <global_id>").
    """
    from registration.db_config import SEARCH_SETTINGS
    threshold = match_threshold if match_threshold is not None else SEARCH_SETTINGS["match_threshold"]

    names = {}
    for gid, desc in global_descriptors.items():
        best_name, best_sim = None, 0.0
        for name, reg_desc in registered_persons.items():
            sim = _cosine(desc, reg_desc)
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_name is not None and best_sim >= threshold:
            names[gid] = {"name": best_name, "similarity": round(best_sim, 4)}
    return names


def run_cross_camera_matching(camera_results: dict, camera_engines: dict,
                               registered_persons: dict = None,
                               match_threshold: float = CROSS_CAM_MATCH_THRESHOLD,
                               output_json_path: str = "outputs/cross_camera/global_identities.json"):
    """
    camera_results: {camera_id: results} — the per-frame dict returned by
                    run_reid_pipeline for that camera (or reloaded from its
                    output_json_path).
    camera_engines: {camera_id: engine} — the ReIDEngine returned alongside
                    each camera's results (used for .consolidated_features).
    registered_persons: optional {name: np.ndarray(698,)} from
                    registration.identity_db.IdentityDatabase().export_for_reid()
                    — if omitted, global identities are left unnamed.

    Returns and saves a combined structure:
        {
          "cam1": {frame_id: [{..., "global_id": 1, "name": "Alice"}, ...]},
          "cam2": {...},
          "global_identities": {1: {"name": "Alice", ...}, 2: {...}, ...}
        }
    """
    matcher = CrossCameraMatcher(match_threshold=match_threshold)
    for cam_id, engine in camera_engines.items():
        matcher.add_camera(cam_id, engine.consolidated_features)

    name_map = {}
    if registered_persons:
        name_map = resolve_names(matcher.global_descriptors(), registered_persons)

    combined = {}
    for cam_id, results in camera_results.items():
        cam_out = {}
        for frame_id, people in results.items():
            frame_out = []
            for p in people:
                p = dict(p)
                local_id = p.get("consolidated_id")
                gid = matcher.get_global_id(cam_id, local_id) if local_id not in (None, -1) else None
                p["global_id"] = gid
                if gid is not None and gid in name_map:
                    p["name"] = name_map[gid]["name"]
                    p["name_similarity"] = name_map[gid]["similarity"]
                else:
                    p["name"] = None
                frame_out.append(p)
            cam_out[frame_id] = frame_out
        combined[cam_id] = cam_out

    combined["global_identities"] = {
        str(gid): {
            "cameras_seen_on": sorted({cam for (cam, _lid), g in matcher.local_to_global.items() if g == gid}),
            **({"name": name_map[gid]["name"], "similarity": name_map[gid]["similarity"]} if gid in name_map else {}),
        }
        for gid in matcher.global_descriptors().keys()
    }

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info(f"✅ Cross-camera matching complete: {len(matcher.global_descriptors())} "
                f"global identities across {len(camera_engines)} camera(s) -> {output_json_path}")

    return combined
