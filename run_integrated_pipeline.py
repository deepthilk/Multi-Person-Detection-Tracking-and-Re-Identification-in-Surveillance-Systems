"""
Final Integration: Multi-Camera -> Wanted-Person Filter -> Per-Camera Re-ID
                    -> Cross-Camera Matching -> Name Resolution -> rendered video
================================================================================

This is the piece that ties every teammate's module together for Phase 3,
without editing any of their files:

  Lekha   (multicamera.multi_cam_pipeline)   - captures + tracks every camera,
                                                writes per-camera tracking.json
                                                in the exact schema reid_main.py
                                                already expects.
  (STEP 1.5) face-based WANTED-PERSON filter - after tracking, every track that
                                                does not confidently match a
                                                registered face is dropped, so
                                                ONLY registered people go on to
                                                be re-identified and named.
  (STEP 1.25) tracker-level crossing/occlusion fix - between tracking and the
                                                wanted filter, corrects DeepSORT
                                                id swaps at crossings using face
                                                evidence (reidentification.
                                                crossing_fix), so the wanted
                                                filter never names a box that
                                                rides the wrong person's body.
  Deepthi (reidentification.reid_main)       - run_reid_pipeline(), called
                                                ONCE PER CAMERA, unmodified.
  Deepthi (reidentification.cross_camera_match) - NEW: merges each camera's
                                                independent stable_ids into
                                                one global_id per real person.
  Prajna  (registration.identity_db)         - export_for_reid() supplies the
                                                {name: embedding} dict used to
                                                attach real names to global ids.
  Pranjali (web/ dashboard)                  - consumes the combined JSON this
                                                script writes.

Usage:
    python run_integrated_pipeline.py --device cuda            # wanted people only
    python run_integrated_pipeline.py --device cpu --track-everyone  # track everyone
    python run_integrated_pipeline.py --device cpu --max-frames 200   # smoke test
"""

import argparse
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reidentification.face_encoder import confirm_distance as _encoder_confirm_distance


def _body_gate_default():
    """Default body-descriptor gate (cosine) for the wanted filter."""
    from reidentification.face_name_resolver import BODY_GATE_THRESHOLD
    return BODY_GATE_THRESHOLD


# A person is only a trustworthy body-gate candidate when their registered body
# images are self-consistent: at least two images whose mean pairwise cosine
# clears this bar. A single screenshot / mixed-outfit / loose-crop anchor is a
# diluted descriptor that sits near everyone and drags strangers through the
# cosine gate (e.g. Lekha's 4 images averaged 0.69 self-sim and 12 stranger
# tracks cleared 0.75). Such persons can still be found by FACE; they are just
# never promoted by body appearance alone.
MIN_BODY_IMAGES = 2
MIN_BODY_SELF_SIM = 0.80


def _trustworthy_body_persons(body_db, body_persons):
    """Drop registered persons whose body embeddings are an unreliable anchor."""
    import numpy as _np
    trusted = {}
    for name, avg in body_persons.items():
        rec = body_db.get_person(name)
        embs = (rec or {}).get("embeddings", [])
        if len(embs) < MIN_BODY_IMAGES:
            logger.info(f"Body gate: '{name}' skipped "
                        f"(only {len(embs)} body image(s), need >= {MIN_BODY_IMAGES})")
            continue
        arr = [_np.asarray(e, dtype=_np.float32) for e in embs]
        sims = []
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                a, b = arr[i], arr[j]
                na, nb = _np.linalg.norm(a), _np.linalg.norm(b)
                if na > 1e-8 and nb > 1e-8:
                    sims.append(float(_np.dot(a, b) / (na * nb)))
        mean_self = float(_np.mean(sims)) if sims else 1.0
        if mean_self < MIN_BODY_SELF_SIM:
            logger.info(f"Body gate: '{name}' skipped (inconsistent body images, "
                        f"self-similarity {mean_self:.3f} < {MIN_BODY_SELF_SIM})")
            continue
        trusted[name] = avg
    return trusted if trusted else None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_TRACKING_DIR = Path("outputs/multicam/tracking_raw")
CACHE_META_PATH = Path("outputs/multicam/.cache_meta.json")
WANTED_NAMES_CACHE = Path("outputs/multicam/wanted_names.json")
REID_DIR = Path("outputs/reid")

# Full-raw Re-ID runs per camera feed the wanted-person classifier with
# appearance identity links (which tracks are the same person). Kept separate
# from REID_DIR (which holds the final per-camera Re-ID over the FILTERED
# tracking) so the two passes never invalidate each other's caches.
RAW_REID_DIR = Path("outputs/reid/raw")

# WANTED_RULE_VERSION: bump whenever the wanted-person classification rule
# changes (e.g. the relaxed-sparse vote, the identity-link promotion, the
# confident-match margin, or the face-first / body-gate default) so stale
# wanted_names caches from older rule versions are never reused. Bumped to 3
# when the STEP 1.25 crossing fix was introduced (it re-keys tracks, so a v2
# wanted_names map referenced ids that no longer exist); bumped to 4 for the
# face-first default + FACE_MATCH_MARGIN rule.
WANTED_RULE_VERSION = 4

# STEP 1.25 crossing-fix marker: {version, mtime}. When the marker matches the
# code version and every raw tracking file predates it, the RAW_TRACKING_DIR
# files are already the corrected ids, so the fix is skipped (it is otherwise
# idempotent but does slow face extraction on every cached re-run).
CROSSING_FIX_VERSION = 1
CROSSING_FIX_MARKER = Path("outputs/multicam/.crossing_fix_meta.json")


def _registration_mtime():
    """Latest mtime across the registration DBs the wanted filter depends on
    (face_db.json for face gating, identity_db.json for the body fallback),
    or 0 if none exist. Any registration/rebuild rewrites one of them, so
    caches computed against a different set of registered people are
    invalidated."""
    mtimes = []
    for path in (
        "outputs/registration/face_db.json",
        "outputs/registration/identity_db.json",
    ):
        try:
            mtimes.append(Path(path).stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else 0


def _source_signatures(cameras):
    """{source: mtime} for cache invalidation. Missing files => None (always miss)."""
    sig = {}
    for cfg in cameras:
        src = cfg["source"]
        if isinstance(src, str) and Path(src).exists():
            sig[src] = Path(src).stat().st_mtime
        else:
            sig[str(src)] = None
    return sig


def _write_step1_meta(args, cameras):
    CACHE_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_META_PATH.write_text(json.dumps({
        "max_frames": args.max_frames,
        "sources": _source_signatures(cameras),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))


def step1_cache_valid(args, cameras):
    """STEP 1 (tracking) cache is valid when the same videos + max_frames as the
    last full run produced outputs/multicam/tracking_raw/<cam>_tracking.json."""
    if args.no_cache:
        return False
    if not CACHE_META_PATH.exists():
        return False
    try:
        meta = json.loads(CACHE_META_PATH.read_text())
    except Exception:
        return False
    if meta.get("max_frames") != args.max_frames:
        return False
    for src, mtime in _source_signatures(cameras).items():
        if mtime is None or meta.get("sources", {}).get(src) != mtime:
            return False
    for cfg in cameras:
        cam_id = cfg["camera_id"]
        if not (RAW_TRACKING_DIR / f"{cam_id}_tracking.json").exists():
            return False
    return True


def _save_raw_tracking(per_camera_tracking):
    """Persist the PRE-filter (raw) tracking so a later run can skip STEP 1 and
    still re-run / reuse the wanted-person filter correctly."""
    for cam_id, tracking in per_camera_tracking.items():
        if not tracking:
            continue
        path = RAW_TRACKING_DIR / f"{cam_id}_tracking.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tracking, indent=2))


def _load_per_camera_tracking(cameras, raw=True):
    per_camera_tracking = {}
    for cfg in cameras:
        cam_id = cfg["camera_id"]
        if raw:
            path = RAW_TRACKING_DIR / f"{cam_id}_tracking.json"
        else:
            path = Path("outputs/multicam/tracking") / f"{cam_id}_tracking.json"
        if not path.exists():
            per_camera_tracking[cam_id] = {}
            continue
        data = json.loads(path.read_text())
        per_camera_tracking[cam_id] = {int(k): v for k, v in data.items()}
    return per_camera_tracking


def _save_wanted_names(wanted_names, threshold, args):
    WANTED_NAMES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if threshold is None:
        threshold = _encoder_confirm_distance()
    body_gate_threshold = (getattr(args, "body_gate_threshold", None)
                           or _body_gate_default())
    WANTED_NAMES_CACHE.write_text(json.dumps({
        "version": WANTED_RULE_VERSION,
        "threshold": threshold,
        "face_db_mtime": _registration_mtime(),
        "body_gate": {
            "disabled": not bool(getattr(args, "use_body_gate", False)),
            "threshold": body_gate_threshold,
        },
        "wanted": [[c, t, r] for (c, t), r in wanted_names.items()],
    }, indent=2, default=str))


def _load_wanted_names(args):
    """Reuse the wanted_names map from a previous run only when the filter was
    computed with the same classification rule version AND face-name threshold
    AND body-gate configuration on the same raw tracking AND the same face DB
    (re-registering / rebuilding the face or identity DB must force the filter
    to re-run, otherwise stale wanted tracks keep being named against people
    who were removed / re-registered)."""
    if args.no_cache or not WANTED_NAMES_CACHE.exists():
        return None
    try:
        data = json.loads(WANTED_NAMES_CACHE.read_text())
    except Exception:
        return None
    if data.get("version") != WANTED_RULE_VERSION:
        return None
    wanted_threshold = (args.face_name_threshold if args.face_name_threshold is not None
                        else _encoder_confirm_distance())
    if data.get("threshold") != wanted_threshold:
        return None
    if data.get("face_db_mtime", -1) != _registration_mtime():
        return None
    body_gate_threshold = (getattr(args, "body_gate_threshold", None)
                           or _body_gate_default())
    if data.get("body_gate", {}).get("disabled") != (not bool(getattr(args, "use_body_gate", False))):
        return None
    if data.get("body_gate", {}).get("threshold") != body_gate_threshold:
        return None
    return {(str(c), str(t)): r for c, t, r in data.get("wanted", [])}


def _reid_cache_valid(cam_id, tracking_json_path, video_src, mode):
    reid_path = REID_DIR / f"{cam_id}_reid_results.json"
    mode_path = REID_DIR / f"{cam_id}_mode.json"
    if not reid_path.exists():
        return False
    if not mode_path.exists() or json.loads(mode_path.read_text()).get("mode") != mode:
        return False
    if not Path(tracking_json_path).exists():
        return False
    if reid_path.stat().st_mtime < Path(tracking_json_path).stat().st_mtime:
        return False
    if isinstance(video_src, str) and Path(video_src).exists():
        if reid_path.stat().st_mtime < Path(video_src).stat().st_mtime:
            return False
    return True


def _write_reid_mode(cam_id, mode):
    mode_path = REID_DIR / f"{cam_id}_mode.json"
    mode_path.parent.mkdir(parents=True, exist_ok=True)
    mode_path.write_text(json.dumps({"mode": mode}))


def _load_reid(cam_id):
    """Rebuild (engine shim, results) from the persisted reid JSON. Cross-camera
    matching only reads engine.consolidated_features + per-frame results, both of
    which are fully persisted there."""
    reid_path = REID_DIR / f"{cam_id}_reid_results.json"
    data = json.loads(reid_path.read_text())
    results = {int(k): v for k, v in data.get("frames", {}).items()}
    engine = SimpleNamespace(consolidated_features={
        int(k): np.asarray(v, dtype=np.float32)
        for k, v in data.get("consolidated_features", {}).items()
    })
    return engine, results


def _raw_reid_cache_valid(cam_id, raw_tracking_json_path, video_src):
    """The full-raw Re-ID pass is valid while its input tracking JSON and the
    source video are unchanged."""
    reid_path = RAW_REID_DIR / f"{cam_id}_reid_results.json"
    if not reid_path.exists():
        return False
    if not Path(raw_tracking_json_path).exists():
        return False
    if reid_path.stat().st_mtime < Path(raw_tracking_json_path).stat().st_mtime:
        return False
    if isinstance(video_src, str) and Path(video_src).exists():
        if reid_path.stat().st_mtime < Path(video_src).stat().st_mtime:
            return False
    return True


def _run_raw_reid(cameras, args):
    """
    Run Re-ID once per camera over the FULL raw tracking (every detected
    track, not just the face-gated subset) purely to get appearance identity
    links for the wanted-person classifier. Returns a flat map
        {(cam_id, track_id_str): consolidated_id}
    (cids are per-camera numbers, so the camera is part of the key). Results
    are cached in RAW_REID_DIR (never the same files as the final per-camera
    Re-ID in STEP 2, so the two passes don't invalidate each other).
    """
    from reidentification.reid_main import run_reid_pipeline

    track_cids = {}
    raw_consolidated = {}
    for cfg in cameras:
        cam_id = cfg["camera_id"]
        raw_tracking_json = RAW_TRACKING_DIR / f"{cam_id}_tracking.json"
        if not raw_tracking_json.exists():
            logger.warning(f"  {cam_id}: no raw tracking - no identity links")
            continue
        reid_path = RAW_REID_DIR / f"{cam_id}_reid_results.json"
        if not args.no_cache and _raw_reid_cache_valid(cam_id, raw_tracking_json, cfg["source"]):
            logger.info(f"[CACHE] {cam_id}: raw Re-ID (wanted classification) reused from {reid_path}")
        else:
            logger.info(f"- {cam_id} raw Re-ID (wanted classification, full tracking) -")
            run_reid_pipeline(
                video_path=cfg["source"],
                tracking_json_path=str(raw_tracking_json),
                output_json_path=str(reid_path),
                device=args.device,
                debug_trace=args.debug_trace,
            )
        data = json.loads(reid_path.read_text())
        raw_consolidated[cam_id] = {
            int(k): np.asarray(v, dtype=np.float32)
            for k, v in data.get("consolidated_features", {}).items()
        }
        per_cam = {}
        for people in data.get("frames", {}).values():
            for p in people:
                cid = p.get("consolidated_id")
                if cid in (None, -1):
                    continue
                per_cam[(cam_id, str(p.get("id")))] = cid
        track_cids.update(per_cam)
        logger.info(f"  {cam_id}: linked {len(per_cam)} track(s) to identities")
    return track_cids, raw_consolidated


def _tracking_id_map(tracking):
    """{(cam_id, track_id, frame): bbox} flat view for cheap change detection."""
    keys = set()
    for cam_id, frames in tracking.items():
        for f, people in frames.items():
            for p in people:
                keys.add((cam_id, int(p["id"]), int(f)))
    return keys


def apply_crossing_fix(per_camera_tracking, cameras, args=None):
    """
    STEP 1.25: tracker-level crossing/occlusion correction (reidentification.
    crossing_fix). Runs on the RAW per-camera tracking right after STEP 1 and
    before the wanted-person filter, so every downstream stage sees the
    corrected track ids. Idempotent: once applied, the raw tracking on disk is
    already corrected and the marker below lets later cached runs skip the
    (slow) face-extraction pass entirely.
    """
    try:
        from reidentification.crossing_fix import correct_crossing_identities
    except ImportError:
        logger.info("STEP 1.25: crossing fix unavailable - skipping")
        return per_camera_tracking

    if args is not None and getattr(args, "no_cache", False):
        skip = False
    else:
        skip = False
        try:
            marker = json.loads(CROSSING_FIX_MARKER.read_text())
            skip = (marker.get("version") == CROSSING_FIX_VERSION
                    and marker.get("face_db_mtime", -1) == _registration_mtime()
                    and all(
                        (RAW_TRACKING_DIR / f"{cfg['camera_id']}_tracking.json").stat().st_mtime
                        <= marker.get("mtime", 0)
                        for cfg in cameras
                        if (RAW_TRACKING_DIR / f"{cfg['camera_id']}_tracking.json").exists()
                    )
                    and all(
                        (RAW_TRACKING_DIR / f"{cfg['camera_id']}_tracking.json").exists()
                        for cfg in cameras
                    ))
        except Exception:
            skip = False
    if skip:
        logger.info("[CACHE] STEP 1.25 crossing fix already applied - reusing corrected ids")
        return per_camera_tracking

    logger.info("=" * 70)
    logger.info("STEP 1.25: Tracker-Level Crossing/Occlusion Fix (id re-association)")
    logger.info("=" * 70)
    original_keys = _tracking_id_map(per_camera_tracking)
    fixed = correct_crossing_identities(per_camera_tracking, cameras)
    if _tracking_id_map(fixed) != original_keys:
        _save_raw_tracking(fixed)
        CROSSING_FIX_MARKER.parent.mkdir(parents=True, exist_ok=True)
        CROSSING_FIX_MARKER.write_text(json.dumps({
            "version": CROSSING_FIX_VERSION,
            "mtime": time.time(),
            "face_db_mtime": _registration_mtime(),
        }))
    else:
        logger.info("  No crossing/occlusion id changes - raw tracking left as-is")
        CROSSING_FIX_MARKER.parent.mkdir(parents=True, exist_ok=True)
        CROSSING_FIX_MARKER.write_text(json.dumps({
            "version": CROSSING_FIX_VERSION,
            "mtime": time.time(),
            "face_db_mtime": _registration_mtime(),
        }))
    return fixed


def apply_face_names(combined, args, wanted_names=None):
    """
    Override body-descriptor name resolution with FACE-based naming (reliable
    on this footage). When `wanted_names` (the authoritative track->name map
    from the wanted-person filter) is supplied, it is stamped directly instead
    of re-sampling faces. Skips silently when the face DB has no usable entries
    or face_recognition is unavailable.
    """
    if args.no_face_names or args.skip_names:
        return combined
    try:
        from reidentification.face_name_resolver import (
            resolve_track_names, stamp_names_from_map, face_validate_gids,
            verify_frame_names,
        )
        from multicamera.camera_config import CAMERAS
        from registration.face_db import FaceIdentityDB

        face_db = FaceIdentityDB()
        if not face_db.list_persons():
            logger.info("Face DB empty - skipping FACE name resolution (register photos + "
                        "run 'register.py face-db --rebuild' to enable)")
            return combined
        logger.info("=" * 70)
        logger.info("STEP 3b: FACE-Based Name Resolution (per track, overrides body names)")
        logger.info("=" * 70)
        if wanted_names:
            combined = stamp_names_from_map(combined, wanted_names)
            combined = face_validate_gids(combined, wanted_names)
            combined = verify_frame_names(
                combined, cameras=CAMERAS, face_db=face_db,
                threshold=args.face_name_threshold,
            )
        else:
            combined = resolve_track_names(
                combined, cameras=CAMERAS, face_db=face_db,
                threshold=args.face_name_threshold,
            )
        import json as _json
        with open(args.output, "w") as _f:
            _json.dump(combined, _f, indent=2, default=str)
        logger.info(f"   Updated output (face names): {args.output}")
    except ImportError:
        logger.warning("face_recognition not installed - FACE name resolution disabled")
    except Exception as e:
        logger.warning(f"[WARN]   FACE name resolution failed ({e}); keeping body names")
    return combined


def filter_wanted_people(per_camera_tracking, args, track_cids=None,
                         raw_consolidated=None):
    """
    STEP 1.5: drop every track that does not confidently match a registered
    (wanted) face. Returns (per_camera_tracking, wanted_names) where
    wanted_names is {(cam_id, track_id): rec}. Everything after this step only
    ever sees wanted people. Falls back to tracking everyone when the face DB
    is empty. `track_cids` (optional full-raw Re-ID identity links) enables
    identity-link promotion of faceless wanted tracks; `raw_consolidated`
    ({cam_id: {cid: descriptor}}) enables the body-descriptor fallback.
    """
    if not args.wanted_only:
        return per_camera_tracking, {}
    try:
        from registration.face_db import FaceIdentityDB
        from reidentification.face_name_resolver import filter_tracking_to_wanted
        from multicamera.camera_config import CAMERAS
        import json as _json
        from multicamera.camera_config import MULTICAM_SETTINGS
        from pathlib import Path

        face_db = FaceIdentityDB()
        if not face_db.list_persons():
            logger.info("Face DB empty - tracking everyone (no wanted-person filter)")
            return per_camera_tracking, {}

        # Body-descriptor fallback is OPT-IN (--use-body-gate). Face-first
        # naming is the default: names come ONLY from face votes + identity
        # links, so a registered person is named whenever they face the camera
        # in any clothes / any video - clothing never decides identity.
        body_persons = None
        use_body_gate = bool(getattr(args, "use_body_gate", False))
        if use_body_gate:
            try:
                from registration.identity_db import IdentityDatabase
                body_db = IdentityDatabase()
                if len(body_db):
                    body_persons = body_db.export_for_reid()
                    body_persons = _trustworthy_body_persons(body_db, body_persons)
                else:
                    body_persons = None
            except Exception as e:
                logger.warning(f"[WARN]   Could not load registered body descriptors ({e})")
        else:
            logger.info("Face-first naming active (body-descriptor fallback disabled - "
                        "use --use-body-gate to enable clothing-based naming)")

        logger.info("=" * 70)
        if use_body_gate:
            logger.info("STEP 1.5: Wanted-Person Filter (face + identity-link + body gated)")
        else:
            logger.info("STEP 1.5: Wanted-Person Filter (face + identity-link, face-first)")
        logger.info("=" * 70)
        filtered, wanted_names = filter_tracking_to_wanted(
            per_camera_tracking, CAMERAS, face_db=face_db,
            threshold=args.face_name_threshold,
            track_cids=track_cids,
            body_persons=body_persons,
            raw_consolidated=raw_consolidated,
            body_gate_threshold=args.body_gate_threshold,
        )
        for cfg in CAMERAS:
            cam_id = cfg["camera_id"]
            if cam_id not in filtered:
                continue
            path = Path(MULTICAM_SETTINGS["tracking_dir"]) / f"{cam_id}_tracking.json"
            with open(path, "w") as f:
                _json.dump(filtered[cam_id], f)

        n_before = sum(len(v) for v in per_camera_tracking.values())
        n_after = sum(len(v) for v in filtered.values())
        logger.info(f"  Wanted tracks: {sorted((k[1], k[0]) for k in wanted_names)}")
        logger.info(f"  Kept {len(wanted_names)} wanted track(s) "
                    f"(tracking frames {n_before} -> {n_after})")
        _save_wanted_names(wanted_names, args.face_name_threshold, args)
        return filtered, wanted_names
    except ImportError:
        logger.warning("No face encoder available - tracking everyone")
        return per_camera_tracking, {}
    except Exception as e:
        logger.warning(f"[WARN]   Wanted-person filter failed ({e}); tracking everyone")
        return per_camera_tracking, {}


def main():
    parser = argparse.ArgumentParser(description="Full multi-camera Detection->Tracking->Re-ID->Cross-Camera pipeline")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--conf-threshold", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Optional cap on multicam ticks (quick smoke test)")
    parser.add_argument("--cross-cam-threshold", type=float, default=0.60,
                         help="Cosine similarity threshold for merging identities across cameras")
    parser.add_argument("--skip-names", action="store_true",
                         help="Skip name resolution even if the registration DB has entries")
    parser.add_argument("--no-face-names", action="store_true",
                         help="Disable FACE-based name resolution (body-descriptor names only)")
    parser.add_argument("--wanted-only", dest="wanted_only", action="store_true", default=True,
                         help="Only track / re-identify people registered in the face DB; "
                              "everyone else is dropped right after tracking (default: on)")
    parser.add_argument("--track-everyone", dest="wanted_only", action="store_false",
                         help="Keep tracking ALL detected people (disable wanted-person gating)")
    parser.add_argument("--face-name-threshold", type=float, default=None,
                         help="Face distance threshold for naming identities "
                              "(default: the active face encoder's confident cutoff, "
                              "0.40 for dlib, ~0.45 for ArcFace)")
    parser.add_argument("--use-body-gate", action="store_true",
                        help="OPT-IN: enable the body-descriptor fallback (names tracks whose "
                             "body Re-ID descriptor matches a registered person when no face "
                             "confirms them). DEFAULT IS FACE-FIRST: names come ONLY from face "
                             "votes + identity links, so clothing never decides identity - a "
                             "registered person is named the moment they face the camera, in "
                             "any clothes, in any video.")
    parser.add_argument("--no-body-gate", action="store_true",
                        help="Legacy alias - the body-descriptor fallback is already disabled "
                             "by default (face-first naming). Kept so old command lines still work.")
    parser.add_argument("--body-gate-threshold", type=float, default=_body_gate_default(),
                         help="Cosine gate for the body-descriptor fallback (higher = stricter, "
                              "default: 0.85). Only persons with >= 2 self-consistent registered "
                              "body images are eligible for body gating.")
    parser.add_argument("--output", type=str, default="outputs/cross_camera/global_identities.json")
    parser.add_argument("--debug-trace", action="store_true",
                         help="Log every ID lock/switch/reappear/new-identity decision with the "
                              "exact scores behind it - grep the output for a frame number "
                              "(frame ≈ seconds_into_video * fps) to see exactly why a swap happened")
    parser.add_argument("--no-cache", action="store_true",
                         help="Recompute STEP 1 (tracking) and STEP 2 (Re-ID) from scratch, "
                              "ignoring cached outputs")
    parser.add_argument("--track-level", action="store_true",
                         help="Use track-level global clustering (reidentification.track_cluster) "
                              "instead of the frame-by-frame ReIDEngine + separate cross-camera "
                              "matching step. Averages descriptors over each whole DeepSORT track "
                              "and clusters once, globally, rather than deciding identity every "
                              "frame in real time - see track_cluster.py's module docstring for why.")
    args = parser.parse_args()

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable - falling back to CPU")
        device = "cpu"

    # ── Step 1: multi-camera capture, detection & tracking (Lekha's module) ──
    logger.info("=" * 70)
    logger.info("STEP 1: Multi-Camera Capture, Detection & Tracking")
    logger.info("=" * 70)
    from multicamera.multi_cam_pipeline import run_multicamera_pipeline
    from multicamera.camera_config import CAMERAS, MULTICAM_SETTINGS

    cache_hit = step1_cache_valid(args, CAMERAS)
    if cache_hit:
        per_camera_tracking = _load_per_camera_tracking(CAMERAS, raw=True)
        logger.info(f"[CACHE] STEP 1 tracking reused from {RAW_TRACKING_DIR} "
                    f"({sum(len(v) for v in per_camera_tracking.values())} "
                    f"tracking frames; videos unchanged)")
    else:
        records, per_camera_tracking = run_multicamera_pipeline(
            device=device, conf_threshold=args.conf_threshold, max_frames=args.max_frames,
        )
        logger.info(f"[OK]  {len(records)} person records across "
                    f"{len(per_camera_tracking)} camera(s)")
        _save_raw_tracking(per_camera_tracking)
        _write_step1_meta(args, CAMERAS)
        logger.info(f"[CACHE] STEP 1 raw tracking cached for fast re-runs")

    per_camera_tracking = apply_crossing_fix(per_camera_tracking, CAMERAS, args)

    cached_wanted = _load_wanted_names(args) if cache_hit else None
    if args.wanted_only and cached_wanted is not None:
        per_camera_tracking = _load_per_camera_tracking(CAMERAS, raw=False)
        wanted_names = cached_wanted
        logger.info(f"[CACHE] wanted-person filter reused "
                    f"({len(wanted_names)} wanted tracks)")
    else:
        track_cids = None
        raw_consolidated = None
        if args.wanted_only:
            track_cids, raw_consolidated = _run_raw_reid(CAMERAS, args)
        per_camera_tracking, wanted_names = filter_wanted_people(
            per_camera_tracking, args, track_cids=track_cids,
            raw_consolidated=raw_consolidated)

    registered_persons = None
    if not args.skip_names:
        try:
            from registration.identity_db import IdentityDatabase
            db = IdentityDatabase()
            if len(db):
                registered_persons = {
                    name: rec["embeddings"]
                    for name, rec in db._data.items()
                }
                logger.info(f"Loaded {len(registered_persons)} registered person(s) ({sum(len(v) for v in registered_persons.values())} total embeddings) for name resolution")
            else:
                logger.info("Registration DB is empty - global identities will be unnamed")
        except Exception as e:
            logger.warning(f"[WARN]   Could not load registration DB ({e}); continuing without names")

    if args.track_level:
        # ── Track-level path: one clustering pass replaces Steps 2+3 ──────
        logger.info("=" * 70)
        logger.info("STEP 2+3: Track-Level Re-Identification & Clustering")
        logger.info("=" * 70)
        from reidentification.track_cluster import run_track_level_pipeline

        per_camera_tracking_paths = {}
        for cfg in CAMERAS:
            cam_id = cfg["camera_id"]
            if cam_id not in per_camera_tracking or not per_camera_tracking[cam_id]:
                logger.warning(f"⏭️  {cam_id}: no tracking data, skipping")
                continue
            per_camera_tracking_paths[cam_id] = f"{MULTICAM_SETTINGS['tracking_dir']}/{cam_id}_tracking.json"

        if not per_camera_tracking_paths:
            logger.error("[FAIL]  No camera produced tracking data - nothing to cluster")
            return 1

        combined = run_track_level_pipeline(
            camera_configs=CAMERAS,
            per_camera_tracking_paths=per_camera_tracking_paths,
            device=device,
            max_frames=args.max_frames,
            registered_persons=registered_persons,
            output_json_path=args.output,
        )
        combined = apply_face_names(combined, args, wanted_names=wanted_names)
        n_global = len(combined.get("global_identities", {}))
        n_named = sum(1 for v in combined["global_identities"].values() if "name" in v)
        logger.info("=" * 70)
        logger.info(f"[OK]  Pipeline complete (track-level): {n_global} global identities "
                    f"({n_named} matched to a registered name)")
        logger.info(f"   Dashboard-ready output: {args.output}")
        logger.info("=" * 70)
        return 0

    # ── Step 2: Re-ID, once per camera, using Deepthi's UNMODIFIED pipeline ──
    logger.info("=" * 70)
    logger.info("STEP 2: Per-Camera Re-Identification")
    logger.info("=" * 70)
    from reidentification.reid_main import run_reid_pipeline

    camera_results = {}
    camera_engines = {}
    for cfg in CAMERAS:
        cam_id = cfg["camera_id"]
        if cam_id not in per_camera_tracking or not per_camera_tracking[cam_id]:
            logger.warning(f"⏭️  {cam_id}: no tracking data, skipping Re-ID")
            continue

        if args.wanted_only:
            tracking_json = f"{MULTICAM_SETTINGS['tracking_dir']}/{cam_id}_tracking.json"
            reid_mode = "wanted_only"
        else:
            tracking_json = f"{RAW_TRACKING_DIR}/{cam_id}_tracking.json"
            reid_mode = "track_everyone"
        reid_output = f"{REID_DIR}/{cam_id}_reid_results.json"
        logger.info(f"- {cam_id} ({cfg['source']}) -")

        if not args.no_cache and _reid_cache_valid(cam_id, tracking_json, cfg["source"], reid_mode):
            engine, results = _load_reid(cam_id)
            logger.info(f"[CACHE] {cam_id}: Re-ID results reused from {reid_output}")
        else:
            engine, results = run_reid_pipeline(
                video_path=cfg["source"],
                tracking_json_path=tracking_json,
                output_json_path=reid_output,
                device=device,
                debug_trace=args.debug_trace,
            )
            _write_reid_mode(cam_id, reid_mode)
        camera_engines[cam_id] = engine
        camera_results[cam_id] = results

    if not camera_engines:
        logger.error("[FAIL]  No camera produced Re-ID results - nothing to cross-match")
        return 1

    # ── Step 3: cross-camera identity matching (NEW - Deepthi's final piece) ─
    logger.info("=" * 70)
    logger.info("STEP 3: Cross-Camera Identity Matching")
    logger.info("=" * 70)
    from reidentification.cross_camera_match import run_cross_camera_matching

    combined = run_cross_camera_matching(
        camera_results=camera_results,
        camera_engines=camera_engines,
        registered_persons=registered_persons,
        match_threshold=args.cross_cam_threshold,
        output_json_path=args.output,
    )
    combined = apply_face_names(combined, args, wanted_names=wanted_names)

    n_global = len(combined.get("global_identities", {}))
    n_named = sum(1 for v in combined["global_identities"].values() if "name" in v)
    logger.info("=" * 70)
    logger.info(f"[OK]  Pipeline complete: {n_global} global identities "
                f"({n_named} matched to a registered name)")
    logger.info(f"   Dashboard-ready output: {args.output}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
