"""
Face-based name resolution for cross-camera global identities.
===============================================================

WHY THIS EXISTS
---------------
Name resolution in the main pipeline (cross_camera_match.py / track_cluster.py)
compares the 698-dim BODY descriptor to the registration DB. That body model is
fine-tuned on ~16 identities and is NOT discriminative enough for reliable name
matching (it is what caused wrong matches when searching by body).

This module re-runs name resolution on the *same* combined output using FACE
encodings instead - authoritative, footage-calibrated (threshold 0.40, and a
global identity is only named when >= MIN_VOTES of its sampled frames agree).
A wrong face match would need to repeat across several independent frames of a
track, which a false positive does not do.

USAGE (as a post-processing step on any pipeline's output JSON)
---------------------------------------------------------------
    from reidentification.face_name_resolver import resolve_global_names
    from multicamera.camera_config import CAMERAS

    combined = resolve_global_names(combined, cameras=CAMERAS)

or via the standalone script:
    python name_global_ids.py --input outputs/cross_camera/global_identities.json

The function mutates AND returns `combined`, writing `name`, `face_similarity`,
`face_votes` and `name_source="face"` into each named global identity and into
the per-frame person entries. Body-based names (if any) are only kept when no
face name is confident.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from reidentification.face_cue import FaceCueExtractor
from reidentification.face_encoder import confirm_distance as _encoder_confirm_distance
from registration.face_db import (
    FaceIdentityDB, face_distance, distance_to_similarity,
    FACE_CONFIRM_DISTANCE, MIN_SEGMENT_FRAMES,
)

logger = logging.getLogger(__name__)

# Confident same-person face-distance cutoff. dlib's 128-dim Euclidean 0.40
# is the historical value (FACE_CONFIRM_DISTANCE); with the ArcFace backend
# the cutoff is the active encoder's (cosine distance ~0.45). Public functions
# default to `None` and resolve to confirm_distance() at call time so the
# threshold always matches the encoder that produced the embeddings.
FACE_NAME_THRESHOLD = FACE_CONFIRM_DISTANCE
MIN_VOTES = 2
MAX_SAMPLES_PER_GID = 16

# Confident-match margin (distance). A person's face may only be used to name a
# track / global identity when they clearly beat the runner-up registered
# person by at least this much. With the multi-anchor gallery (augmented
# registration photos) this is what makes relaxing the match threshold safe:
# a slightly-loosened threshold only helps genuinely close cross-context
# matches, while any track that sits between two registered people stays
# unnamed instead of being guessed.
FACE_MATCH_MARGIN = 0.05

# A single face just above `threshold` is trusted for a face-sparse track.
# On a track where faces almost never extract (someone far from / turning
# away from the camera), voting cannot accumulate, so one near-threshold
# match is meaningful evidence rather than noise - the same-person band is
# ~0.31 and 0.40 was cutting straight through the wanted people on this
# footage. Only applies to tracks with <= SPARSE_FACE_LIMIT usable faces;
# tracks with plenty of faces must clear min_votes normally.
RELAXED_SPARSE_MARGIN = 0.03
SPARSE_FACE_LIMIT = 4

# Domain-gap tolerance for RELATIVE matching. A face is a confident match when
# its BEST registered-person distance is below `threshold` (absolute match) OR
# when it beats the runner-up registered person by FACE_MATCH_MARGIN while
# staying within `threshold + RELATIVE_MATCH_CEILING`. The second clause is
# what makes surveillance footage match a high-quality registration gallery:
# the same person's embedding is systematically pushed 0.1-0.3 above its
# studio-photo value (low-res, angle, lighting), but it stays clearly CLOSER
# to its own identity than to any other registered person. Using the relative
# margin instead of a fixed absolute cutoff lets a genuinely present person be
# named across a domain gap, while a stranger (who is ~equidistant from every
# registered person) still fails because no clear winner exists.
RELATIVE_MATCH_CEILING = 0.30

# A global identity (appearance link from full-raw Re-ID) may "promote" its
# faceless tracks to a registered name only when at least this many of its
# tracks independently face-confirm the SAME name. This is what lets a
# re-entering wanted person with no extractable faces (e.g. cam1 t10) stay
# boxed, while a contaminated identity (cam2 cid1/cid2 mix wanted people
# with unknowns) never drags an unknown track into a name.
MIN_PROMOTE_ANCHORS = 2

# Body-descriptor fallback: when a track's face evidence is too weak to name it
# but its body appearance (from the full-raw Re-ID pass) closely matches a
# REGISTERED body embedding in identity_db, the track is still kept + named.
# Cosine similarity gate - deliberately strict so an unknown person's body
# almost never clears it. Body descriptors are dominated by clothing
# colour/texture, so anything below ~0.85 lets strangers in similar outfits
# through (the pipeline's own identity-merge threshold is 0.72).
BODY_GATE_THRESHOLD = 0.85


def _bbox(entry):
    """Extract [x1,y1,x2,y2] from a person entry (list or dict forms)."""
    b = entry.get("bbox") if isinstance(entry, dict) else None
    if b is None:
        return None
    if isinstance(b, (list, tuple)):
        if len(b) < 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in b[:4]]
        except (TypeError, ValueError):
            return None
        return [int(x1), int(y1), int(x2), int(y2)]
    if isinstance(b, dict):
        try:
            return [int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])]
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _collect_samples(combined, max_per_gid=MAX_SAMPLES_PER_GID):
    """
    Walk the per-camera sections of a combined output and collect, for every
    global id, up to `max_per_gid` (camera, frame, bbox) samples spread across
    its appearance. Returns ({gid: [(cam, frame, bbox)]}, {cam: {frames}}).
    """
    samples = {}
    frame_sets = {}
    for cam_id, cam_data in combined.items():
        if cam_id == "global_identities" or not isinstance(cam_data, dict):
            continue
        for frame_str, people in cam_data.items():
            try:
                frame = int(frame_str)
            except (TypeError, ValueError):
                continue
            if not isinstance(people, list):
                continue
            for p in people:
                gid = p.get("global_id") if isinstance(p, dict) else None
                if gid is None:
                    continue
                b = _bbox(p)
                if b is None:
                    continue
                samples.setdefault(gid, []).append((cam_id, frame, b))
                frame_sets.setdefault(cam_id, set()).add(frame)

    # keep evenly-spaced samples per global id so a short false-match burst
    # cannot dominate the vote
    out = {}
    for gid, lst in samples.items():
        lst.sort(key=lambda s: (s[0], s[1]))
        if len(lst) <= max_per_gid:
            out[gid] = lst
        else:
            idx = np.linspace(0, len(lst) - 1, max_per_gid).round().astype(int)
            out[gid] = [lst[i] for i in idx]
    return out, frame_sets


def _cam_to_video(cameras):
    """Map every plausible per-camera section key to its video source.
    Handles both 'cam1' and 'video1' style keys in the combined output."""
    import re
    mapping = {}
    for c in cameras:
        cid = c["camera_id"]
        src = str(c["source"])
        mapping[cid] = src
        m = re.search(r"(\d+)$", cid)
        if m:
            mapping[f"video{m.group(1)}"] = src
            mapping[f"camera{m.group(1)}"] = src
    return mapping


def _source_available(video):
    """A source is available when it is a real file, OR an image-sequence
    pattern (e.g. 'demo/clipa/%04d.png') whose first frame exists."""
    v = Path(video)
    if v.exists():
        return True
    if "%" in video:
        return Path(video % 1).exists()
    return False


def _extract_faces(combined, cameras, max_per_gid=MAX_SAMPLES_PER_GID):
    """Extract one face encoding per sampled (camera, frame, bbox) and return
    {gid: [encodings]}."""
    cam_video = _cam_to_video(cameras)
    samples, frame_sets = _collect_samples(combined, max_per_gid)
    if not samples:
        return {}

    extractor = FaceCueExtractor(upsample_times=1)
    gid_faces = {gid: [] for gid in samples}

    by_cam_frame = {}
    for gid, lst in samples.items():
        for cam_id, frame, bbox in lst:
            by_cam_frame.setdefault((cam_id, frame), []).append((gid, bbox))

    for cam_id, frames in frame_sets.items():
        video = cam_video.get(cam_id)
        if not video or not _source_available(video):
            logger.warning(f"  {cam_id}: video not found ({video}) - cannot face-name it")
            continue
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            logger.warning(f"  {cam_id}: could not open {video}")
            continue
        try:
            for frame in sorted(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
                ok, img = cap.read()
                if not ok:
                    continue
                for gid, bbox in by_cam_frame.get((cam_id, frame), []):
                    enc = extractor.extract(img, bbox)
                    if enc is not None:
                        gid_faces[gid].append(enc)
        finally:
            cap.release()
    return gid_faces


def _effective_threshold(threshold):
    """Use the ACTIVE encoder's confident distance cutoff when the caller did
    not pass one explicitly (so the threshold always matches the embeddings)."""
    return _encoder_confirm_distance() if threshold is None else threshold


def resolve_global_names(combined, cameras, face_db=None,
                         threshold=None,
                         min_votes=MIN_VOTES,
                         max_encodings_per_gid=MAX_SAMPLES_PER_GID):
    """
    Attach real names to the global identities in `combined` by matching FACE
    encodings sampled from each identity's track against the registered face DB.

    Args:
        combined: the pipeline output dict (per-camera per-frame results +
                  combined["global_identities"]).
        cameras:  list of {"camera_id": ..., "source": video_path} (camera_config.CAMERAS).
        face_db:  FaceIdentityDB (defaults to outputs/registration/face_db.json).

    Returns:
        combined (mutated in place) with `name`/`name_source="face"` fields.
    """
    face_db = face_db or FaceIdentityDB()
    persons = face_db.list_persons()
    if not persons:
        logger.info("Face DB empty - skipping face-based name resolution")
        return combined

    logger.info("Face-based name resolution:")
    logger.info(f"  Registered persons: {len(persons)} | threshold={threshold} "
                f"min_votes={min_votes}")

    gid_faces = _extract_faces(combined, cameras, max_encodings_per_gid)
    if not gid_faces:
        logger.info("  No face encodings could be extracted for any global identity")
        return combined

    faces_used = sum(len(v) for v in gid_faces.values())
    logger.info(f"  Sampled {faces_used} face encoding(s) across {len(gid_faces)} global identity(ies)")

    names = {}
    for gid, encs in gid_faces.items():
        if not encs:
            continue
        votes = {}
        best_dist = {}
        for name in persons:
            q = face_db.encodings_for(name)
            if not q:
                continue
            ds = [d for d in (face_distance(q, e) for e in encs)
                  if d is not None]
            if not ds:
                continue
            votes[name] = sum(1 for d in ds if d <= threshold)
            best_dist[name] = min(ds)
        if not votes:
            continue
        best = max(votes, key=lambda n: votes[n])
        if votes[best] < min_votes:
            continue
        best_d = best_dist[best]
        # Margin rule: the winner must clearly beat the runner-up, otherwise
        # the global identity sits between two registered people (ambiguous).
        runner_d = min(d for n, d in best_dist.items() if n != best) \
            if len(best_dist) > 1 else None
        if runner_d is not None and best_d + FACE_MATCH_MARGIN > runner_d:
            continue
        names[gid] = {
            "name": best,
            "similarity": round(distance_to_similarity(best_d), 4),
            "face_distance": round(best_d, 4),
            "face_votes": votes[best],
            "name_source": "face",
        }
        logger.info(f"  gid {gid}: '{best}' (votes {votes[best]}/{len(encs)}, "
                    f"best distance {best_d:.3f})")

    if not names:
        logger.info("  No global identity confidently matched a registered face "
                    f"(need >= {min_votes} agreeing frames)")
        return combined

    gids = combined.setdefault("global_identities", {})
    for gid, nm in names.items():
        key = str(gid)
        if key in gids and isinstance(gids[key], dict):
            gids[key].update(nm)
        else:
            gids[key] = {"cameras_seen_on": [], **nm}

    # Body-derived names that CONTRADICT a face-confirmed name are cleared so the
    # final output has zero false positives: if global identity X was confidently
    # named "PersonA" by face, no other global identity may keep a body-only
    # "PersonA" label unless face evidence independently confirmed it.
    rejected_gids = set()
    face_names = {nm["name"] for nm in names.values()}
    for gid_key, info in gids.items():
        if not isinstance(info, dict) or info.get("name_source") == "face":
            continue
        if info.get("name") not in face_names:
            continue
        try:
            gid_int = int(gid_key)
        except (TypeError, ValueError):
            continue
        if not gid_faces.get(gid_int):
            continue  # could not face-examine -> leave the body guess alone
        rejected_gids.add(gid_int)
        stale_name = info["name"]
        info["name"] = None
        info.pop("similarity", None)
        info.pop("name_similarity", None)
        info["name_source"] = "face_rejected"
        logger.info(f"  gid {gid_int}: cleared body name '{stale_name}' "
                    "(face evidence rejects it)")

    for cam_id, cam_data in combined.items():
        if cam_id == "global_identities" or not isinstance(cam_data, dict):
            continue
        for people in cam_data.values():
            if not isinstance(people, list):
                continue
            for p in people:
                if not isinstance(p, dict):
                    continue
                gid = p.get("global_id")
                if gid in names:
                    p["name"] = names[gid]["name"]
                    p["name_similarity"] = names[gid]["similarity"]
                    p["name_source"] = "face"
                elif gid in rejected_gids:
                    p["name"] = None
                    p.pop("name_similarity", None)
                    p["name_source"] = "face_rejected"

    n_named = sum(1 for v in gids.values() if v.get("name_source") == "face")
    logger.info(f"  {n_named}/{len(gids)} global identity(ies) named by face")
    return combined


# ──────────────────────────────────────────────────────────────────────────
# TRACK-LEVEL naming
# --------------------------------------------------------------------------
# Body-based cross-camera clustering mixes different people into one global
# identity on this footage, so naming at the gid level mislabels some frames
# (and misses Pranjali, whose track landed in an unnamed gid). The reliable
# unit of evidence is the DeepSORT track: the user identifies people per
# track, and face evidence per track is clean. Names are therefore attached
# per (camera, track) and only rolled up to gids afterwards.
# ──────────────────────────────────────────────────────────────────────────


def _collect_track_samples(combined, max_per_track=MAX_SAMPLES_PER_GID):
    """{(cam_id, track_id): [(frame, bbox)]} sampled evenly per track."""
    track_samples = {}
    for cam_id, cam_data in combined.items():
        if cam_id == "global_identities" or not isinstance(cam_data, dict):
            continue
        for frame_str, people in cam_data.items():
            try:
                frame = int(frame_str)
            except (TypeError, ValueError):
                continue
            if not isinstance(people, list):
                continue
            for p in people:
                if not isinstance(p, dict):
                    continue
                tid = p.get("id")
                b = _bbox(p)
                if tid is None or b is None:
                    continue
                track_samples.setdefault((cam_id, str(tid)), []).append((frame, b))

    out = {}
    for key, lst in track_samples.items():
        lst.sort()
        if len(lst) <= max_per_track:
            out[key] = lst
        else:
            idx = np.linspace(0, len(lst) - 1, max_per_track).round().astype(int)
            out[key] = [lst[i] for i in idx]
    return out


def _extract_track_faces(combined, cameras, max_per_track=MAX_SAMPLES_PER_GID):
    """{track_key: [encodings]} - one encoding per sampled (frame, bbox)."""
    cam_video = _cam_to_video(cameras)
    samples = _collect_track_samples(combined, max_per_track)
    if not samples:
        return {}

    extractor = FaceCueExtractor(upsample_times=1)
    track_faces = {key: [] for key in samples}

    by_cam_frame = {}
    frame_sets = {}
    for (cam_id, tid), lst in samples.items():
        frame_sets.setdefault(cam_id, set()).update(f for f, _ in lst)
        for frame, bbox in lst:
            by_cam_frame.setdefault((cam_id, frame), []).append(((cam_id, tid), bbox))

    for cam_id, frames in frame_sets.items():
        video = cam_video.get(cam_id)
        if not video or not _source_available(video):
            logger.warning(f"  {cam_id}: video not found ({video}) - cannot face-name it")
            continue
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            logger.warning(f"  {cam_id}: could not open {video}")
            continue
        try:
            for frame in sorted(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
                ok, img = cap.read()
                if not ok:
                    continue
                for track_key, bbox in by_cam_frame.get((cam_id, frame), []):
                    enc = extractor.extract(img, bbox)
                    if enc is not None:
                        track_faces[track_key].append(enc)
        finally:
            cap.release()
    return track_faces


def _frame_winner(ds, threshold):
    """The registered person this frame's face clearly belongs to, or None.

    `ds` maps registered name -> face distance for one sampled face. A face
    votes for its closest registered person only when that person is a CLEAR
    winner: it beats the runner-up by at least FACE_MATCH_MARGIN AND its
    distance is within `threshold` (absolute match) or `threshold +
    RELATIVE_MATCH_CEILING` (relative match, for the registration-vs-surveillance
    domain gap). A stranger is roughly equidistant from every registered person,
    so no clear winner exists and the face votes for no one.
    """
    if not ds:
        return None
    best = min(ds, key=ds.get)
    best_d = ds[best]
    runner_d = min(d for n, d in ds.items() if n != best) if len(ds) > 1 else None
    if runner_d is not None and best_d + FACE_MATCH_MARGIN >= runner_d:
        return None  # ambiguous: no clear winner over the runner-up
    if best_d <= threshold or best_d <= threshold + RELATIVE_MATCH_CEILING:
        return best
    return None


def _vote_track_names(track_faces, face_db, persons, threshold, min_votes):
    """{track_key: {"name","votes","faces","best_distance","similarity"}}.

    Voting is per-face RELATIVE (see _frame_winner): a face only counts for the
    registered person it clearly beats the competition for. This is what names
    a wanted person filmed under a domain shift (their embedding sits above the
    absolute threshold but is unmistakably closer to themselves than to anyone
    else registered), while a stranger - equidistant from everyone - is never
    able to accumulate votes.
    """
    relaxed_sparse_dist = threshold + RELAXED_SPARSE_MARGIN
    track_names = {}
    for key, encs in track_faces.items():
        if not encs:
            continue
        votes = {}
        best_dist = {}
        for enc in encs:
            ds = {}
            for name in persons:
                q = face_db.encodings_for(name)
                if not q:
                    continue
                d = face_distance(q, enc)
                if d is not None:
                    ds[name] = d
            winner = _frame_winner(ds, threshold)
            if winner is not None:
                votes[winner] = votes.get(winner, 0) + 1
                if winner not in best_dist or ds[winner] < best_dist[winner]:
                    best_dist[winner] = ds[winner]
        if not votes:
            continue
        best = max(best_dist, key=lambda n: (votes.get(n, 0), -best_dist[n]))
        best_d = best_dist[best]
        # Margin rule on the best per-person distance: the winner must clearly
        # beat the runner-up registered person, otherwise the track sits between
        # two people and is ambiguous.
        runner_d = None
        for n in best_dist:
            if n == best:
                continue
            if runner_d is None or best_dist[n] < runner_d:
                runner_d = best_dist[n]
        if runner_d is not None and best_d + FACE_MATCH_MARGIN > runner_d:
            continue
        # A single near-threshold match is trusted for face-sparse tracks (few
        # faces ever extract -> voting can't accumulate). Tracks with plenty of
        # faces must clear min_votes normally.
        strong = (len(encs) <= SPARSE_FACE_LIMIT and best_d <= relaxed_sparse_dist)
        if votes.get(best, 0) >= min_votes or strong:
            track_names[key] = {
                "name": best,
                "votes": votes.get(best, 0),
                "faces": len(encs),
                "best_distance": round(best_d, 4),
                "similarity": round(distance_to_similarity(best_d), 4),
            }
    return track_names


def _promote_by_identity(track_names, track_cids, track_faces, face_db, persons,
                         threshold, min_anchors=MIN_PROMOTE_ANCHORS):
    """
    Appearance-link promotion: tracks that share a full-raw Re-ID identity
    (cid) with >= `min_anchors` face-confirmed tracks of the same registered
    name are promoted to that name, provided their own face evidence does not
    contradict it. This recovers a re-entering wanted person whose face never
    extracts (e.g. cam1 t10: 0 faces, but appearance cid1 already contains two
    face-confirmed Deepthi tracks). It deliberately requires >= min_anchors
    independent face confirmations so a contaminated identity (one wanted track
    merged with unknowns, as on cam2) cannot name an unknown track.

    Re-ID consolidated ids are per-camera (renumbered 1..N inside each camera),
    so the identity key is (camera_id, cid) - a raw cid number alone collides
    across cameras and would wrongly merge unrelated people.
    """
    from collections import defaultdict
    cid_tracks = defaultdict(list)
    for (cam_id, tid), cid in track_cids.items():
        cid_tracks[(cam_id, cid)].append((cam_id, tid))

    promoted = 0
    for (cam_id, cid), keys in cid_tracks.items():
        confirmed = defaultdict(int)
        for key in keys:
            rec = track_names.get(key)
            if rec:
                confirmed[rec["name"]] += 1
        if not confirmed:
            continue
        best_name = max(confirmed, key=lambda n: confirmed[n])
        if confirmed[best_name] < min_anchors:
            continue
        for key in keys:
            if key in track_names:
                continue
            if not _face_compatible_with(key, best_name, track_faces, face_db,
                                         persons, threshold):
                continue
            encs = track_faces.get(key, [])
            best_d = votes = None
            if encs:
                q = face_db.encodings_for(best_name)
                ds = [d for d in (face_distance(q, e) for e in encs) if d is not None]
                if ds:
                    best_d = min(ds)
                    votes = sum(1 for d in ds if d <= threshold)
            track_names[key] = {
                "name": best_name,
                "votes": votes or 0,
                "faces": len(encs),
                "best_distance": round(best_d, 4) if best_d is not None else None,
                "similarity": round(distance_to_similarity(best_d), 4) if best_d is not None else None,
            }
            promoted += 1
            logger.info(f"  promoted t{key[1]} ({key[0]}) -> '{best_name}' "
                        f"(same Re-ID identity {cam_id}/cid{cid}, {confirmed[best_name]} face-confirmed)")
    if promoted:
        logger.info(f"  Identity-link promotion named {promoted} track(s) "
                    f"(>= {min_anchors} face-confirmed anchors per identity)")
    return track_names


def _face_compatible_with(track_key, name, track_faces, face_db, persons, threshold):
    """A track with no faces is trivially compatible. A track with faces is
    compatible only if none of them confidently match a DIFFERENT registered
    person (a confident match to another name would veto the promotion - two
    different people can't share one appearance identity)."""
    encs = track_faces.get(track_key, [])
    if not encs:
        return True
    for other in persons:
        if other == name:
            continue
        q = face_db.encodings_for(other)
        ds = [d for d in (face_distance(q, e) for e in encs) if d is not None]
        if ds and min(ds) <= threshold:
            return False
    return True


def resolve_track_names(combined, cameras, face_db=None,
                        threshold=None, min_votes=MIN_VOTES,
                        max_encodings_per_track=MAX_SAMPLES_PER_GID):
    """
    Attach real names to individual DeepSORT tracks (per camera) by FACE
    matching, then roll names up to global identities from their member tracks.

    The per-frame person entries already carry the DeepSORT track id, so this
    is the granularity the user can vouch for (they identify people by track /
    crop). Naming at this level never mislabels a frame because it is a
    different person: a track is one real person (or the evidence is too weak
    to name at all).

    Returns combined (mutated in place):
      - per-frame people get name / name_similarity / name_source="face" when
        their track matched, otherwise name=None (body names cleared).
      - global_identities entries get a name only if every confidently named
        member track agrees.
    """
    face_db = face_db or FaceIdentityDB()
    threshold = _effective_threshold(threshold)
    persons = face_db.list_persons()
    if not persons:
        logger.info("Face DB empty - skipping face-based name resolution")
        return combined

    logger.info("Track-level face-based name resolution:")
    logger.info(f"  Registered persons: {len(persons)} | threshold={threshold} "
                f"min_votes={min_votes}")

    track_faces = _extract_track_faces(combined, cameras, max_encodings_per_track)
    if not track_faces:
        logger.info("  No face encodings could be extracted for any track")
        return combined

    n_faces = sum(len(v) for v in track_faces.values())
    logger.info(f"  Sampled {n_faces} face encoding(s) across {len(track_faces)} track(s)")

    track_names = _vote_track_names(track_faces, face_db, persons, threshold, min_votes)

    # Apply per-frame names at the track level (and clear body names on every
    # other track so no false label survives).
    track_to_gid = {}
    for cam_id, cam_data in combined.items():
        if cam_id == "global_identities" or not isinstance(cam_data, dict):
            continue
        for people in cam_data.values():
            if not isinstance(people, list):
                continue
            for p in people:
                if not isinstance(p, dict):
                    continue
                key = (cam_id, str(p.get("id")))
                if key in track_names:
                    nm = track_names[key]
                    p["name"] = nm["name"]
                    p["name_similarity"] = nm["similarity"]
                    p["name_source"] = "face"
                elif key in track_faces and track_faces.get(key):
                    # Face encodings existed but none matched a registered
                    # person -> the (cleared) body name was actively rejected.
                    p["name"] = None
                    p.pop("name_similarity", None)
                    p["name_source"] = "face_rejected"
                else:
                    # No face ever extracted -> no evidence either way.
                    p["name"] = None
                    p.pop("name_similarity", None)
                    p["name_source"] = "unconfirmed"
                track_to_gid.setdefault(key, set()).add(p.get("global_id"))

    # Roll up to gid level: a gid is named only if its named member tracks all
    # agree on one person.
    gids = combined.setdefault("global_identities", {})
    for gid_key, info in gids.items():
        if not isinstance(info, dict):
            continue
        member_names = set()
        best_rec = None
        for (cam_id, tid), gset in track_to_gid.items():
            if gid_key not in {str(g) for g in gset}:
                continue
            rec = track_names.get((cam_id, tid))
            if rec:
                member_names.add(rec["name"])
                if best_rec is None or rec["votes"] > best_rec["votes"]:
                    best_rec = rec
        if len(member_names) == 1 and best_rec is not None:
            info["name"] = best_rec["name"]
            info["similarity"] = best_rec["similarity"]
            info["face_distance"] = best_rec["best_distance"]
            info["face_votes"] = best_rec["votes"]
            info["name_source"] = "face"
        else:
            info["name"] = None
            info.pop("similarity", None)
            info.pop("face_distance", None)
            info.pop("face_votes", None)
            info["name_source"] = "face_rejected" if len(member_names) > 1 else None

    n_named = sum(1 for v in gids.values() if v.get("name_source") == "face")
    n_tracks = len(track_names)
    logger.info(f"  {n_tracks} track(s) named | {n_named}/{len(gids)} global identity(ies)")
    for key, rec in sorted(track_names.items()):
        logger.info(f"    {key[0]} track {key[1]}: '{rec['name']}' "
                    f"(votes {rec['votes']}/{rec['faces']}, best distance {rec['best_distance']:.3f})")
    return combined


# ──────────────────────────────────────────────────────────────────────────
# WANTED-PERSON GATING
# --------------------------------------------------------------------------
# "Only track and re-identify the registered people." The multicam stage has
# to detect and track EVERYONE first (it needs to see their faces to know who
# is wanted), but right after tracking we drop every track that does not
# confidently match a registered face. Everything downstream (Re-ID, cross-
# camera matching, naming, rendering) then only ever sees the wanted people,
# so no one else gets a box, an identity, or a name.
# ──────────────────────────────────────────────────────────────────────────


def wanted_track_names(tracking_by_cam, cameras, face_db=None,
                       threshold=None, min_votes=MIN_VOTES,
                       max_encodings_per_track=MAX_SAMPLES_PER_GID,
                       track_cids=None, body_persons=None,
                       raw_consolidated=None, body_gate_threshold=BODY_GATE_THRESHOLD):
    """
    Sample faces along every DeepSORT track and return {(cam_id, track_id): rec}
    for the tracks that CONFIDENTLY match a registered person. Uses the same
    voting rules as resolve_track_names (>= min_votes agreeing frames below
    `threshold`, or a single near-threshold match on a sparse track).

    When `track_cids` = {(cam_id, track_id): cid} (full-raw Re-ID identity
    links) is supplied, tracks in an identity that already has >=
    MIN_PROMOTE_ANCHORS face-confirmed tracks of one name are promoted to it
    (see _promote_by_identity) - this recovers re-entering wanted people whose
    faces never extract, without dragging unknown tracks into a name.

    When `body_persons` = {name: 698-d body embedding} (identity_db) and
    `raw_consolidated` = {cam_id: {cid: 698-d descriptor}} (from the full-raw
    Re-ID pass) are supplied, tracks whose faces were extracted but did NOT
    confidently name a person are additionally classified by their BODY
    appearance against the registered body embeddings. This keeps a wanted
    person whose registered face photos don't match the footage (different
    pose/lighting/quality) but whose body matches their registered body images.

    `tracking_by_cam` is the multicam per-camera tracking schema:
        {cam_id: {frame_str: [{"id": track_id, "bbox": [x1,y1,x2,y2]}]}}
    """
    face_db = face_db or FaceIdentityDB()
    threshold = _effective_threshold(threshold)
    persons = face_db.list_persons()
    if not persons:
        logger.info("  Face DB empty - no tracks can be classified as wanted")
        return {}
    track_faces = _extract_track_faces(tracking_by_cam, cameras, max_encodings_per_track)
    track_names = _vote_track_names(track_faces, face_db, persons, threshold, min_votes)
    if track_cids:
        track_names = _promote_by_identity(
            track_names, track_cids, track_faces, face_db, persons, threshold)
    if body_persons and raw_consolidated:
        track_names = _body_gate(track_names, track_faces, track_cids, raw_consolidated,
                                 body_persons, body_gate_threshold)
    return track_names


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


def _body_gate(track_names, track_faces, track_cids, raw_consolidated,
               body_persons, body_gate_threshold):
    """
    Body-descriptor fallback: name (and thereby keep) tracks whose face
    evidence was too weak to name them, using the track's appearance from the
    full-raw Re-ID pass against the REGISTERED body embeddings.

    Only applies to tracks that (a) had a face extracted (they are real visible
    people, not tracker noise) but (b) were NOT face-confirmed, and (c) whose
    face evidence did NOT confidently point at a DIFFERENT registered person.
    The body match must clear a strict cosine gate so unknown people are not
    dragged in via a diluted / contaminated Re-ID identity.
    """
    added = 0
    for (cam_id, tid), faces in track_faces.items():
        key = (cam_id, str(tid))
        if key in track_names or not faces:
            continue
        cid = track_cids.get(key) if track_cids else None
        if cid is None:
            continue
        desc = (raw_consolidated.get(cam_id) or {})
        if str(cid) in desc:
            desc = desc[str(cid)]
        elif cid in desc:
            desc = desc[cid]
        else:
            continue
        if desc is None:
            continue
        best_name, best_sim = None, 0.0
        for name, emb in body_persons.items():
            sim = _cosine(desc, emb)
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_name is not None and best_sim >= body_gate_threshold:
            track_names[key] = {
                "name": best_name,
                "votes": 0,
                "faces": len(faces),
                "best_distance": None,
                "similarity": round(best_sim, 4),
                "source": "body",
            }
            added += 1
            logger.info(f"  Body gate: {cam_id} t{tid} -> '{best_name}' "
                        f"(body similarity {best_sim:.3f} >= {body_gate_threshold})")
    if added:
        logger.info(f"  Body gate named {added} track(s) by body appearance")
    return track_names


def filter_tracking_to_wanted(tracking_by_cam, cameras, face_db=None,
                              threshold=None, min_votes=MIN_VOTES,
                              max_encodings_per_track=MAX_SAMPLES_PER_GID,
                              track_cids=None, body_persons=None,
                              raw_consolidated=None,
                              body_gate_threshold=BODY_GATE_THRESHOLD):
    """
    Remove every track that is not confidently a registered person from the
    per-camera tracking data. Returns (filtered_tracking_by_cam, track_names)
    where `track_names` is the authoritative {(cam_id, track_id): rec} map.
    Kept records also get their real `name` / `name_source="face"` attached.

    `track_cids` (optional full-raw Re-ID identity links) enables identity-link
    promotion of faceless wanted tracks; `body_persons` + `raw_consolidated`
    enable the body-descriptor fallback - see wanted_track_names.
    """
    track_names = wanted_track_names(
        tracking_by_cam, cameras, face_db=face_db,
        threshold=threshold, min_votes=min_votes,
        max_encodings_per_track=max_encodings_per_track,
        track_cids=track_cids, body_persons=body_persons,
        raw_consolidated=raw_consolidated,
        body_gate_threshold=body_gate_threshold,
    )

    filtered = {}
    for cam_id, data in tracking_by_cam.items():
        keep = {}
        for frame, people in data.items():
            if not isinstance(people, list):
                continue
            kept = []
            for p in people:
                if not isinstance(p, dict):
                    continue
                rec = track_names.get((cam_id, str(p.get("id"))))
                if rec is not None:
                    p["name"] = rec["name"]
                    p["name_similarity"] = rec["similarity"]
                    p["name_source"] = _rec_source(rec)
                    kept.append(p)
            if kept:
                keep[frame] = kept
        filtered[cam_id] = keep
    return filtered, track_names


def stamp_names_from_map(combined, track_names):
    """
    Stamp the authoritative {(cam_id, track_id): rec} name map onto a combined
    output (already filtered to wanted tracks) without re-sampling faces.
    Per-frame people get name / name_similarity / name_source="face", and each
    global identity is named when its member tracks all agree on one person.
    """
    track_to_gid = {}
    for cam_id, cam_data in combined.items():
        if cam_id == "global_identities" or not isinstance(cam_data, dict):
            continue
        for people in cam_data.values():
            if not isinstance(people, list):
                continue
            for p in people:
                if not isinstance(p, dict):
                    continue
                key = (cam_id, str(p.get("id")))
                if key in track_names:
                    nm = track_names[key]
                    p["name"] = nm["name"]
                    p["name_similarity"] = nm["similarity"]
                    p["name_source"] = _rec_source(nm)
                    p["name_votes"] = nm.get("votes")
                track_to_gid.setdefault(key, set()).add(p.get("global_id"))

    gids = combined.setdefault("global_identities", {})
    for gid_key, info in gids.items():
        if not isinstance(info, dict):
            continue
        member_names = set()
        best_rec = None
        for (cam_id, tid), gset in track_to_gid.items():
            if gid_key not in {str(g) for g in gset}:
                continue
            rec = track_names.get((cam_id, tid))
            if rec:
                member_names.add(rec["name"])
                if best_rec is None or rec["votes"] > best_rec["votes"]:
                    best_rec = rec
        if len(member_names) == 1 and best_rec is not None:
            info["name"] = best_rec["name"]
            info["similarity"] = best_rec["similarity"]
            info["face_distance"] = best_rec["best_distance"]
            info["face_votes"] = best_rec["votes"]
            info["name_source"] = _rec_source(best_rec)
        elif info.get("name_source") not in ("face", "body"):
            # Member tracks disagree (mixed cluster) -> the body-derived name,
            # if any, would be a false positive. Clear it.
            info["name"] = None
            info.pop("similarity", None)
            info.pop("name_similarity", None)
            info.pop("face_distance", None)
            info.pop("face_votes", None)
            info["name_source"] = "face_conflict" if member_names else None
    return combined


def face_validate_gids(combined, track_names):
    """
    Face-driven correction of the cross-camera grouping.

    Cross-camera matching groups tracks by BODY descriptor, which the body
    model cannot reliably separate between two different registered people
    (Lekha vs Pranjali score ~0.90 - higher than some same-person matches).
    This pass re-derives the grouping from the trusted per-track names (face
    votes + body gate) that were already stamped onto the frames:

      1. a global identity whose member tracks all agree on one person is
         named from that person (fixing the body-derived/conflict state),
      2. a global identity whose member tracks point at MORE THAN ONE person
         is SPLIT into one identity per person, renumbering the per-frame
         global_id so downstream rendering / search only ever sees
         single-person identities,
      3. member tracks with no confident name follow the strongest group of
         their original identity,
      4. every identity that resolved to the SAME registered person is then
         merged into a single identity, so each wanted person appears exactly
         once in the final global identity list (with the cameras they were
         seen on).

    Returns `combined` (mutated in place).
    """
    cam_data = {
        k: v for k, v in combined.items()
        if k != "global_identities" and isinstance(v, dict)
    }

    # ---- collect each track's name rec + current gid from the frames ----
    track_rec = {}
    track_gid = {}
    for cam_id, frames in cam_data.items():
        for people in frames.values():
            if not isinstance(people, list):
                continue
            for p in people:
                if not isinstance(p, dict):
                    continue
                key = (cam_id, str(p.get("id")))
                track_gid[key] = str(p.get("global_id"))
                if key not in track_rec and key in track_names:
                    track_rec[key] = track_names[key]

    # ---- group tracks by gid ----
    gid_tracks = {}
    for key, gid in track_gid.items():
        gid_tracks.setdefault(gid, []).append(key)

    # ---- decide splits and names for every gid ----
    gids = combined.get("global_identities") or {}
    max_id = max((int(k) for k in gids if str(k).isdigit()), default=0)
    remap = {}          # (cam_id, tid) -> new gid (int)
    new_names = {}      # gid (str) -> {"name": str, "rec": rec}
    splits = 0
    for gid, keys in gid_tracks.items():
        by_name = {}
        for key in keys:
            rec = track_rec.get(key)
            if rec:
                by_name.setdefault(rec["name"], []).append(rec)
        if not by_name:
            continue
        if len(by_name) == 1:
            name = next(iter(by_name))
            best = max(by_name[name], key=_rec_strength)
            new_names[gid] = {"name": name, "rec": best}
            continue
        # mixed cluster -> one new identity per person, strongest group first
        groups = sorted(
            by_name.items(),
            key=lambda kv: max(_rec_strength(r) for r in kv[1]),
            reverse=True,
        )
        leader = None
        for name, recs in groups:
            new_id = str(max_id + 1)
            max_id += 1
            best = max(recs, key=_rec_strength)
            new_names[new_id] = {"name": name, "rec": best}
            if leader is None:
                leader = new_id
            for key in keys:
                rec = track_rec.get(key)
                if rec and rec["name"] == name:
                    remap[key] = int(new_id)
        # tracks without a confident name follow the strongest group
        for key in keys:
            if key not in remap:
                remap[key] = int(leader)
        splits += 1
        logger.info(f"  Face-validate: split gid {gid} into {len(groups)} "
                    f"identity group(s) ({', '.join(g for g, _ in groups)})")

    # ---- final gid per track (after splits + merges) ----
    final_gid = {
        key: remap.get(key, int(track_gid[key]))
        for key in track_gid
    }

    # ---- merge every identity that resolved to the same person ----
    merges = 0
    name_keys = {}
    for key, rec in track_rec.items():
        name_keys.setdefault(rec["name"], []).append((key, rec))
    for name, entries in name_keys.items():
        if len(entries) < 2:
            continue
        best_key, best_rec = max(entries, key=lambda kv: _rec_strength(kv[1]))
        target = final_gid[best_key]
        others = {
            final_gid[key] for key, _ in entries if final_gid[key] != target
        }
        if not others:
            continue
        merges += 1
        for key, _ in entries:
            final_gid[key] = target
        new_names[str(target)] = {"name": name, "rec": best_rec}
        logger.info(f"  Face-validate: merged {len(others) + 1} identity "
                    f"group(s) for '{name}' into gid {target}")

    # ---- rewrite per-frame global_id + name_source ----
    for cam_id, frames in cam_data.items():
        for people in frames.values():
            if not isinstance(people, list):
                continue
            for p in people:
                if not isinstance(p, dict):
                    continue
                key = (cam_id, str(p.get("id")))
                new_gid = final_gid.get(key)
                if new_gid is not None:
                    p["global_id"] = new_gid
                    rec = track_rec.get(key)
                    if rec:
                        p["name_source"] = _rec_source(rec)

    # ---- rebuild the global_identities summary from the frames ----
    memberships = {}
    for cam_id, frames in cam_data.items():
        for people in frames.values():
            if not isinstance(people, list):
                continue
            for p in people:
                if not isinstance(p, dict):
                    continue
                gid = str(p.get("global_id"))
                memberships.setdefault(gid, set()).add(cam_id)
    new_gids = {}
    for gid, cams in memberships.items():
        entry = {"cameras_seen_on": sorted(cams), "observations": len(cams)}
        named = new_names.get(gid)
        if named:
            rec = named["rec"]
            entry.update({
                "name": named["name"],
                "name_similarity": rec.get("similarity"),
                "face_distance": rec.get("best_distance"),
                "face_votes": rec.get("votes"),
                "name_source": _rec_source(rec),
            })
        new_gids[gid] = entry
    combined["global_identities"] = new_gids
    if splits or merges:
        logger.info(f"  Face-validate: corrected cross-camera grouping "
                    f"({splits} mixed identity group(s) split, "
                    f"{merges} same-person merge(s))")
    return combined


def verify_frame_names(combined, cameras, face_db=None, threshold=None):
    """Per-frame face verification of stamped track names.

    `stamp_names_from_map` writes ONE track-level name onto EVERY frame of the
    track, so a contaminated track (DeepSORT glued two people into one id) or a
    brief ID-swap (DeepSORT exchanged two nearby boxes for a frame or two)
    labels the wrong person on those frames. This pass re-extracts the face
    from every named frame's bounding box and, per frame, decides on the
    REGISTERED-PERSON MARGIN (who beats whom by how much), not on a fixed
    absolute threshold:

      * stamped person is the best match (clear winner)     -> keep the name,
      * a DIFFERENT registered person clearly wins          -> re-label the frame,
      * no registered person clearly wins (ambiguous face)  -> clear the name
        (a stranger / swapped-in face sitting inside a named track).

    The winner must beat the runner-up registered person by FACE_MATCH_MARGIN
    (and stay within the absolute or domain-gap band, see _frame_winner); a
    frame where the top two registered people are within FACE_MATCH_MARGIN of
    each other is too ambiguous to trust, exactly like the track-level margin
    rule. This keeps a low-quality-but-genuine face (same person, best by ~0.3
    even above the absolute threshold) while removing a stranger face (nobody
    clearly wins).

    Frames whose face cannot be extracted are left untouched - a wanted person
    momentarily looking away is still that person, and the track-level vote
    already decided the name.

    Returns `combined` (mutated in place).
    """
    face_db = face_db or FaceIdentityDB()
    threshold = _effective_threshold(threshold)
    persons = face_db.list_persons()
    if not persons:
        return combined
    cam_video = _cam_to_video(cameras)
    extractor = FaceCueExtractor(upsample_times=1)

    by_cam_frame = {}
    frame_sets = {}
    for cam_id, cam_data in combined.items():
        if cam_id == "global_identities" or not isinstance(cam_data, dict):
            continue
        for frame_str, people in cam_data.items():
            if not isinstance(people, list):
                continue
            for p in people:
                if not isinstance(p, dict) or not p.get("name"):
                    continue
                bbox = p.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                by_cam_frame.setdefault((cam_id, frame_str), []).append((p, bbox))
                frame_sets.setdefault(cam_id, set()).add(frame_str)

    cleared = relabelled = kept = 0
    for cam_id, frames in frame_sets.items():
        video = cam_video.get(cam_id)
        if not video or not _source_available(video):
            logger.warning(f"  {cam_id}: video not found ({video}) - cannot verify frames")
            continue
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            logger.warning(f"  {cam_id}: could not open {video}")
            continue
        try:
            for frame_str in sorted(frames, key=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_str) - 1)
                ok, img = cap.read()
                if not ok:
                    continue
                for p, bbox in by_cam_frame[(cam_id, frame_str)]:
                    enc = extractor.extract(img, bbox)
                    if enc is None:
                        continue  # cannot verify -> keep the track name
                    stamped = p.get("name")
                    if stamped not in persons:
                        continue
                    ds = {}
                    for name in persons:
                        q = face_db.encodings_for(name)
                        if q:
                            d = face_distance(q, enc)
                            if d is not None:
                                ds[name] = d
                    if not ds:
                        continue
                    # The frame is decided by the same RELATIVE rule the track
                    # voter uses: a face is a confident match only when its best
                    # registered person clearly beats the runner-up (within the
                    # absolute or domain-gap band). This keeps a genuine face
                    # filmed under a domain shift (best by ~0.3, sits above the
                    # absolute threshold) AND clears a stranger (roughly
                    # equidistant from every registered person) whose box was
                    # stamped with a name only because the track was contaminated.
                    winner = _frame_winner(ds, threshold)
                    if winner is None:
                        p["name"] = None
                        p.pop("name_similarity", None)
                        p.pop("name_votes", None)
                        p["name_source"] = "unconfirmed"
                        cleared += 1
                        logger.info(f"  {cam_id} frame {frame_str}: cleared "
                                    f"'{stamped}' (ambiguous face, no clear winner)")
                        continue
                    if winner == stamped:
                        kept += 1
                        p["name_similarity"] = round(
                            distance_to_similarity(ds[winner]), 4)
                        continue
                    p["name"] = winner
                    p["name_similarity"] = round(distance_to_similarity(ds[winner]), 4)
                    p["name_source"] = "face"
                    p["name_votes"] = 1
                    relabelled += 1
                    logger.info(f"  {cam_id} frame {frame_str}: re-labelled "
                                f"'{stamped}' -> '{winner}' (face {ds[winner]:.3f})")
        finally:
            cap.release()
    if cleared or relabelled:
        logger.info(f"  Per-frame verify: {kept} frame(s) kept, "
                    f"{cleared} cleared, {relabelled} re-labelled")
    return combined


def _rec_strength(rec):
    """Ordering key for track name recs: more votes first, then similarity."""
    return (rec.get("votes") or 0, rec.get("similarity") or 0.0)


def _rec_source(rec):
    return "body" if rec.get("source") == "body" else "face"
