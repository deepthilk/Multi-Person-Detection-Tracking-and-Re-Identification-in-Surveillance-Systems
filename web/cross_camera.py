"""
Cross-Camera Identity Unification (web pipeline)
=================================================

Each camera's Re-ID run resolves identities independently, so the SAME person
can end up with different track IDs — and different auto-resolved names — on
different cameras (the user's web run: cam2 merged lekha+deeps into "lekha"
while cam1 had deeps correct). This module combines the per-camera results
using the face galleries the web pipeline persists in each reid.json's
"__tracks__" section (512-dim InsightFace ArcFace, the same face space
registration and search use).

Rules:
  * A strong cross-camera face match (max pairwise face sim >= FACE_SAME)
    proves two tracks are the same person, even across different cameras.
  * When faces are missing on a side, only a very strong track-average
    appearance match (>= APPEARANCE_SAME) links them.
  * When faces exist but are inconclusive (FACE_VETO <= sim < FACE_SAME),
    a same-person-level appearance match (>= AMBIGUOUS_APPEARANCE_SAME)
    links them — the best-confidence candidate wins, so a person seen from
    behind / with no readable face on one camera is still matched by what
    they wear.
  * Two tracks on the SAME camera are different people by construction, so a
    linked component can never contain two nodes from one camera.
  * Name conflicts within a component are resolved in favour of a manually
    corrected name, then the highest-confidence auto name; the winner is
    propagated to every track in the component so all cameras agree.

Pure functions only — server.py owns persistence and re-rendering.
"""

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Same-person face similarity measured >= 0.53 on this footage (deeps 0.536,
# lekha 0.576); different people <= 0.28. 0.45 is comfortably between.
FACE_SAME = 0.45
# Different-person faces sit <= 0.28; a face pair below this vetoes a
# cross-camera link even when the bodies look identical.
FACE_VETO = 0.28
# Appearance-only fallback for pairs missing faces on at least one side.
# Body appearance cannot separate lookalikes (lekha<->deeps app ~0.75-0.80),
# so the bar stays high; genuine reappearances measure >= 0.84.
APPEARANCE_SAME = 0.75
# Appearance bar for pairs whose faces exist but are INCONCLUSIVE
# (FACE_VETO <= face_sim < FACE_SAME) — e.g. a low-quality / out-of-view face
# that neither confirms nor rules out a match. In non-uniform footage the
# appearance cue separates people cleanly (measured here: different people
# ~0.12-0.25, same person ~0.85-0.97), so a SAME-PERSON-LEVEL appearance match
# is allowed to close the link. Matches AMBIGUOUS_FACE_APPEARANCE_SAME in
# reid_main.py.
AMBIGUOUS_APPEARANCE_SAME = 0.85


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def _max_pair_face_sim(faces_a, faces_b) -> float:
    if not faces_a or not faces_b:
        return None
    best = 0.0
    for fa in faces_a:
        fa = np.asarray(fa, dtype=np.float32)
        for fb in faces_b:
            fb = np.asarray(fb, dtype=np.float32)
            sim = _cosine(fa, fb)
            if sim > best:
                best = sim
    return best


def _track_similarity(track, name, cameras: dict) -> tuple:
    """Best evidence a track is `name`, given OTHER tracks (on any camera)
    already resolved to that name. Returns (similarity, face_sim) using
    cross-camera face/ appearance evidence so propagated names carry a real
    confidence instead of None."""
    refs = [
        t
        for cam_tracks in cameras.values()
        for t in cam_tracks.values()
        if t.get("name") == name and t is not track
    ]
    if not refs:
        return track.get("similarity"), track.get("face_sim")

    best_face = None
    best_app = None
    for ref in refs:
        fs = _max_pair_face_sim(track.get("faces", []), ref.get("faces", []))
        if fs is not None and (best_face is None or fs > best_face):
            best_face = fs
        ma, mb = track.get("mean_feature"), ref.get("mean_feature")
        if ma and mb:
            app = _cosine(ma, mb)
            if best_app is None or app > best_app:
                best_app = app
    if best_face is not None and best_face >= FACE_SAME:
        return round(best_face, 3), round(best_face, 3)
    if best_app is not None:
        return round(best_app, 3), None
    return track.get("similarity"), track.get("face_sim")


def unify_cameras(cameras: dict) -> dict:
    """
    cameras: {camera_id: {final_track_id: track_dict}}
             where track_dict has keys faces (list of 512-dim lists),
             mean_feature (list), name, similarity, face_sim, cues, manual.

    Returns: {camera_id: {final_track_id: {name, similarity, face_sim,
             cues, global_id}}} with every track's resolved name propagated
             consistently across cameras. Unchanged tracks are still listed
             (with their existing name) so callers can apply unconditionally.
    """
    # (camera_id, final_track_id) -> component id
    parent = {}
    cam_of = {}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    nodes = []
    for cam_id, tracks in cameras.items():
        for tid, track in tracks.items():
            node = (cam_id, str(tid))
            parent[node] = node
            cam_of[node] = cam_id
            nodes.append(node)

    def merge(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # A component may never hold two tracks from the same camera.
        cams_ra = {cam_of[n] for n in nodes if find(n) == ra}
        cams_rb = {cam_of[n] for n in nodes if find(n) == rb}
        if cams_ra & cams_rb:
            return
        parent[rb] = ra

    # Collect every candidate cross-camera link with a confidence score, then
    # merge greedily by trust tier so the STRONGEST link wins. Face-confirmed
    # pairs (>= FACE_SAME) are the most trustworthy and are applied first —
    # behaviour unchanged. Pairs whose faces are inconclusive (FACE_VETO <=
    # sim < FACE_SAME) fall back to a same-person-level body-appearance match
    # (>= AMBIGUOUS_APPEARANCE_SAME): appearance separates people in
    # non-uniform footage even when the face is unreadable, and ranking by
    # confidence ensures the best candidate on the other camera wins instead
    # of whichever pair happens to be visited first. Appearance-only pairs
    # (faces missing on a side) keep their existing, slightly lower bar.
    # Face-disagree pairs (sim < FACE_VETO) never link on appearance.
    links = []
    cam_ids = list(cameras.keys())
    for i, cam_a in enumerate(cam_ids):
        for cam_b in cam_ids[i + 1:]:
            for tid_a, track_a in cameras[cam_a].items():
                for tid_b, track_b in cameras[cam_b].items():
                    node_a = (cam_a, str(tid_a))
                    node_b = (cam_b, str(tid_b))
                    face_sim = _max_pair_face_sim(track_a.get("faces", []), track_b.get("faces", []))
                    ma = track_a.get("mean_feature")
                    mb = track_b.get("mean_feature")
                    if face_sim is not None:
                        if face_sim >= FACE_SAME:
                            links.append((0, face_sim, node_a, node_b))
                        elif face_sim < FACE_VETO:
                            continue   # faces clearly disagree -> never link on appearance
                        elif ma and mb:
                            app = _cosine(ma, mb)
                            if app >= AMBIGUOUS_APPEARANCE_SAME:
                                links.append((1, app, node_a, node_b))
                    else:
                        if ma and mb and _cosine(ma, mb) >= APPEARANCE_SAME:
                            links.append((2, _cosine(ma, mb), node_a, node_b))

    links.sort(key=lambda x: (x[0], -x[1]))
    for tier, score, node_a, node_b in links:
        merge(node_a, node_b)

    components = {}
    for node in nodes:
        components.setdefault(find(node), []).append(node)

    corrected = {}
    for cam_id in cam_ids:
        corrected[cam_id] = {}

    next_gid = 1
    for nodes in components.values():
        gid = next_gid
        next_gid += 1
        if len(nodes) < 2:
            node = nodes[0]
            cam_id, tid = node
            t = cameras[cam_id][str(tid)]
            corrected[cam_id][str(tid)] = {
                "name": t.get("name"),
                "similarity": t.get("similarity"),
                "face_sim": t.get("face_sim"),
                "cues": t.get("cues", []),
                "global_id": None if len(cam_ids) > 1 else gid,
            }
            continue
        # Choose the component name: manual correction wins, else the
        # highest-confidence auto-resolved name; empty/None is a name too.
        best = None
        for cam_id, tid in nodes:
            t = cameras[cam_id][str(tid)]
            if t.get("manual"):
                best = t
                break
            if not t.get("name"):
                continue
            if best is None or not best.get("name"):
                best = t
                continue
            if (t.get("similarity") or 0) > (best.get("similarity") or 0):
                best = t
        name = best.get("name") if best else None
        if len(nodes) > 1:
            named = [c for c in nodes if cameras[c[0]][str(c[1])].get("name")]
            distinct = {cameras[c[0]][str(c[1])].get("name") for c in named}
            if len(distinct) > 1:
                logger.warning(
                    "Cross-camera name conflict %s -> resolved to '%s' (auto names: %s)",
                    [f"{c[0]}#{c[1]}" for c in nodes], name, sorted(distinct),
                )

        for cam_id, tid in nodes:
            track = cameras[cam_id][str(tid)]
            sim, face_sim = track.get("similarity"), track.get("face_sim")
            if track.get("name") != name:
                sim, face_sim = _track_similarity(track, name, cameras) if name else (None, None)
            cues = track.get("cues", [])
            if name and "face" not in cues and face_sim is not None:
                cues = list(cues) + ["face"]
            corrected[cam_id][str(tid)] = {
                "name": name,
                "similarity": sim,
                "face_sim": face_sim,
                "cues": cues,
                "global_id": gid,
            }

    return corrected


def unify_session_files(reid_paths: dict) -> dict:
    """
    Convenience wrapper for server.py: reid_paths = {camera_id: reid_json_path}.
    Loads each file's "__tracks__", runs unify_cameras, and returns the same
    corrected structure as unify_cameras.
    """
    cameras = {}
    for cam_id, path in reid_paths.items():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tracks = data.get("__tracks__", {}) if isinstance(data, dict) else {}
        cameras[cam_id] = {str(t): tdata for t, tdata in tracks.items()}
    return unify_cameras(cameras)
