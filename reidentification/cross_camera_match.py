"""
Cross-Camera Identity Matching
================================

Matches identities discovered independently per camera into a single global ID
space. The ReIDEngine within each camera already ensures identity consistency
- this module links identities ACROSS cameras.

STRATEGY
--------
1. Per-camera features (698-dim, same descriptor space) are collected from
   each camera's ReIDEngine.consolidated_features.
2. A similarity matrix between local and global identities is computed.
3. Hungarian assignment finds the globally optimal matching.
4. Unmatched local identities become new global identities.
5. Global descriptors are updated as running averages across cameras.

CROSS-CAMERA CHALLENGES
-----------------------
- Different camera angles -> different body appearance
- Different lighting -> different color histograms
- The 698-dim descriptor relies on deep features (512-dim) which are
  lighting/angle invariant, but color/texture cues are not.
- Threshold needs to be lower than intra-camera since cross-view variation
  is higher than within-view variation.
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

CROSS_CAM_MATCH_THRESHOLD = 0.85


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


class CrossCameraMatcher:
    def __init__(self, match_threshold: float = CROSS_CAM_MATCH_THRESHOLD):
        self.match_threshold = match_threshold
        self._global_descriptors: dict = {}
        self._next_global_id = 1
        self.local_to_global: dict = {}
        self._camera_counts: dict = {}
        self._match_history: dict = {}

    def _to_np(self, desc):
        return np.asarray(desc, dtype=np.float32)

    def add_camera(self, camera_id: str, consolidated_features: dict):
        local_ids = list(consolidated_features.keys())
        if not local_ids:
            return

        if not self._global_descriptors:
            for lid in local_ids:
                gid = self._next_global_id
                self._next_global_id += 1
                self._global_descriptors[gid] = {
                    "descriptor": self._to_np(consolidated_features[lid]).copy(),
                    "cameras": [camera_id],
                    "count": 1
                }
                self.local_to_global[(camera_id, lid)] = gid
            return

        global_ids = list(self._global_descriptors.keys())
        cost = np.zeros((len(local_ids), len(global_ids)), dtype=np.float32)
        sim_matrix = np.zeros((len(local_ids), len(global_ids)), dtype=np.float32)
        for i, lid in enumerate(local_ids):
            local_vec = self._to_np(consolidated_features[lid])
            for j, gid in enumerate(global_ids):
                sim = _cosine(local_vec, self._global_descriptors[gid]["descriptor"])
                sim_matrix[i, j] = sim
                cost[i, j] = 1.0 - sim

        row_idx, col_idx = linear_sum_assignment(cost)
        matched_local = set()

        for r, c in zip(row_idx, col_idx):
            sim = sim_matrix[r, c]
            if sim >= self.match_threshold:
                lid, gid = local_ids[r], global_ids[c]
                self.local_to_global[(camera_id, lid)] = gid
                n = self._global_descriptors[gid]["count"]
                alpha = 1.0 / (n + 1)
                local_vec = self._to_np(consolidated_features[lid])
                new_desc = (1.0 - alpha) * self._global_descriptors[gid]["descriptor"] + alpha * local_vec
                new_desc /= (np.linalg.norm(new_desc) + 1e-8)
                self._global_descriptors[gid]["descriptor"] = new_desc
                self._global_descriptors[gid]["count"] += 1
                if camera_id not in self._global_descriptors[gid]["cameras"]:
                    self._global_descriptors[gid]["cameras"].append(camera_id)
                matched_local.add(lid)

        for lid in local_ids:
            if lid in matched_local:
                continue
            gid = self._next_global_id
            self._next_global_id += 1
            self._global_descriptors[gid] = {
                "descriptor": self._to_np(consolidated_features[lid]).copy(),
                "cameras": [camera_id],
                "count": 1
            }
            self.local_to_global[(camera_id, lid)] = gid

    def get_global_id(self, camera_id: str, local_id: int):
        return self.local_to_global.get((camera_id, local_id))

    def global_descriptors(self) -> dict:
        return {gid: info["descriptor"] for gid, info in self._global_descriptors.items()}

    def global_metadata(self) -> dict:
        return {
            gid: {
                "cameras_seen_on": info["cameras"],
                "observations": info["count"],
            }
            for gid, info in self._global_descriptors.items()
        }


def resolve_names(global_descriptors: dict, registered_persons: dict,
                   match_threshold: float = None) -> dict:
    from registration.db_config import SEARCH_SETTINGS
    threshold = match_threshold if match_threshold is not None else SEARCH_SETTINGS["match_threshold"]

    names = {}
    for gid, desc in global_descriptors.items():
        best_name, best_sim = None, 0.0
        for name, reg_data in registered_persons.items():
            if isinstance(reg_data, list):
                for emb in reg_data:
                    sim = _cosine(desc, emb)
                    if sim > best_sim:
                        best_name, best_sim = name, sim
            else:
                sim = _cosine(desc, reg_data)
                if sim > best_sim:
                    best_name, best_sim = name, sim
        if best_name is not None and best_sim >= threshold:
            names[gid] = {"name": best_name, "similarity": round(best_sim, 4)}
    return names


def run_cross_camera_matching(camera_results: dict, camera_engines: dict,
                                registered_persons: dict = None,
                                match_threshold: float = CROSS_CAM_MATCH_THRESHOLD,
                                output_json_path: str = "outputs/cross_camera/global_identities.json"):
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

    meta = matcher.global_metadata()
    combined["global_identities"] = {
        str(gid): {
            "cameras_seen_on": info["cameras_seen_on"],
            "observations": info["observations"],
            **({"name": name_map[gid]["name"], "similarity": name_map[gid]["similarity"]} if gid in name_map else {}),
        }
        for gid, info in meta.items()
    }

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info(f"[OK]  Cross-camera matching complete: {len(meta)} global identities "
                f"across {len(camera_engines)} camera(s) -> {output_json_path}")

    return combined
