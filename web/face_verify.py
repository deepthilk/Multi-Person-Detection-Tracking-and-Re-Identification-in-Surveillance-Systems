"""
Face-first identity verification & swap correction (post-pass)
==============================================================

Why this exists
---------------
The Re-ID engine's per-frame ids come from a body-appearance + face blend.
When people wear IDENTICAL uniforms, body appearance carries no identity
information (everyone scores 0.85-0.99 against each other), so the engine
can produce three classic failures the user actually sees:

  1. WRONG NAME   — a track is labelled with the appearance coin-flip
                    winner ("Pranjali gets deeps as id"), instead of the
                    person their FACE actually matches.
  2. FALSE MERGE  — two different people who never overlap are merged into
                    one identity by appearance (e.g. trackers 3+9 -> one
                    "deeps" sid).
  3. TRACKER SWAP — a DeepSORT id follows person A then switches to
                    person B mid-video (or two ids exchange people after a
                    crossing), so per-frame boxes carry the wrong identity.

Faces (InsightFace ArcFace, 512-dim, same space as the registration
galleries) stay distinctive in identical uniforms, so this pass re-asserts
identity from FACE evidence only:

  • classify every box's face against the identity DB (stride-sampled),
  • rebuild each tracker's per-frame identity timeline,
  • split a tracker into identity segments wherever its face confidently
    changes person,
  • detect and correct reciprocal swaps between concurrent trackers,
  • resolve each identity's NAME by majority face vote (with a margin),
  • rewrite reid.json (per-frame consolidated ids + __tracks__) and return
    a corrected people summary.

It is deliberately conservative: it only changes a box's identity or a
track's name when the face evidence is decisive (score and vote margins);
otherwise it leaves the engine's output untouched.
"""

import json
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# InsightFace ArcFace scale measured on this project's footage:
# same person >= ~0.54, different people <= ~0.28.
CONFIDENT_FACE = 0.50      # a face this strong is trusted to name a box
STRONG_FACE = 0.58         # a face this strong can SPLIT a tracker
SWAP_SCORE_MIN = 0.50      # reciprocal switch needs both sides this confident
MIN_VOTES = 3              # fewest confident faces before we trust a name
MARGIN_VOTES = 2           # top person must beat the runner-up by this many votes
# Max gap (in frames) between a tracker's segment switch and the reciprocal
# tracker's switch before they count as the same swap event.
SWAP_WINDOW = 30
# A tracker's faces this far below another tracker's faces are treated as a
# DIFFERENT person for the one-sided false-merge split (same-person >= ~0.54,
# different-person <= ~0.28 on this footage, so 0.40 is a safe midpoint).
_SPLIT_DIFFERENT_FACE = 0.40


def _face_sim(a, b) -> float:
    from reidentification.insight_face import InsightFaceExtractor
    return float(InsightFaceExtractor.similarity(
        np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))


class _FaceClassifier:
    """Caches DB galleries and classifies query faces -> (name, score)."""

    def __init__(self, db):
        self.gallery = {}
        for name in db.list_persons():
            rec = db.get_person(name)
            faces = rec.get("face_embeddings") or []
            if faces:
                self.gallery[name] = [np.asarray(f, dtype=np.float32) for f in faces]

    def classify(self, face):
        """Return (best_name, best_score) or (None, 0.0) when no gallery."""
        if face is None or not self.gallery:
            return None, 0.0
        best_name, best_score = None, 0.0
        for name, gfaces in self.gallery.items():
            s = max(_face_sim(face, g) for g in gfaces)
            if s > best_score:
                best_name, best_score = name, s
        return best_name, best_score


def _collect_faces(video_path, tracking_data, frame_ids, stride, extractor):
    """Extract a face for every box in tracking_data at `stride`, aligned to
    `frame_ids`. Returns {frame: {tid: face}}."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("face_verify: cannot open video %s", video_path)
        return {}

    step = max(1, int(stride))
    out = {}
    cur = 0
    for fid in frame_ids:
        if (fid - 1) % step != 0:
            continue
        if cur + 1 < fid:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fid - 1))
            cur = fid - 1
        ok, frame = cap.read()
        cur = fid
        if not ok or frame is None:
            continue
        for t in tracking_data.get(str(fid), []):
            tid = t.get("id")
            bbox = t.get("bbox")
            if tid is None or not bbox:
                continue
            try:
                face = extractor.extract(frame, bbox)
            except Exception:
                face = None
            if face is not None:
                out.setdefault(fid, {})[tid] = face
    cap.release()
    return out


def _cid_at(results, fid, tid):
    for p in results.get(str(fid), []):
        if p.get("id") == tid:
            return p.get("consolidated_id")
    return None


def _detect_and_fix_swaps(classified, results):
    """
    Correct reciprocal tracker swaps: two concurrent trackers that exchange
    identities (A reads X-then-Y while B reads Y-then-X, switches within
    SWAP_WINDOW). Their per-frame consolidated ids are exchanged from the
    switch boundary onward. Returns the number of swaps fixed.
    """
    # reduce each tracker's confident face sequence to stretches
    stretches = {}
    for fid, boxes in sorted(classified.items()):
        for tid, (name, score) in boxes.items():
            if name is None or score < SWAP_SCORE_MIN:
                continue
            runs = stretches.setdefault(tid, [])
            if runs and runs[-1][2] == name and fid - runs[-1][1] <= SWAP_WINDOW:
                runs[-1][1] = fid
            else:
                runs.append([fid, fid, name])

    tids = sorted(stretches, key=int)
    fixed = 0
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            ra, rb = stretches[a], stretches[b]
            if len(ra) < 2 or len(rb) < 2:
                continue
            if ra[0][2] == ra[1][2] or rb[0][2] == rb[1][2]:
                continue
            xa, ya = ra[0][2], ra[1][2]
            xb, yb = rb[0][2], rb[1][2]
            if xa != yb or ya != xb:
                continue
            fa, fb = ra[1][0], rb[1][0]
            if abs(fa - fb) > SWAP_WINDOW:
                continue
            boundary = min(fa, fb)
            _exchange_after(results, a, b, boundary)
            fixed += 1
            logger.info("face_verify: SWAP tracker %s <-> %s at frame %s (%s<->%s)",
                        a, b, boundary, xa, ya)
    return fixed


def _exchange_after(results, tid_a, tid_b, boundary):
    for fid_str, people in results.items():
        if fid_str.startswith("_"):
            continue
        fid = int(fid_str)
        if fid < boundary:
            continue
        c_a = c_b = None
        for p in people:
            if p.get("id") == tid_a:
                c_a = p.get("consolidated_id")
            elif p.get("id") == tid_b:
                c_b = p.get("consolidated_id")
        if c_a is None or c_b is None:
            continue
        for p in people:
            if p.get("id") == tid_a:
                p["consolidated_id"] = c_b
            elif p.get("id") == tid_b:
                p["consolidated_id"] = c_a


def _split_person_switches(classified, results):
    """
    Split a tracker whose STRONG faces show a clean person change with no
    reciprocal counterpart (a DeepSORT id that stopped following one person
    and picked up another). The segment after the switch is minted a fresh
    identity. Returns the number of splits.
    """
    stretches = {}
    for fid, boxes in sorted(classified.items()):
        for tid, (name, score) in boxes.items():
            if name is None or score < STRONG_FACE:
                continue
            runs = stretches.setdefault(tid, [])
            if runs and runs[-1][2] == name and fid - runs[-1][1] <= SWAP_WINDOW:
                runs[-1][1] = fid
            else:
                runs.append([fid, fid, name])

    splits = 0
    seeds = {}
    for tid, runs in stretches.items():
        if len(runs) < 2:
            continue
        first_name = runs[0][2]
        changed = [r for r in runs[1:] if r[2] != first_name]
        if not changed:
            continue
        boundary = changed[0][0]
        old_cid = _cid_at(results, min(r[0] for r in runs), tid)
        if old_cid is None:
            continue
        new_cid = _new_cid(results)
        for fid_str, people in results.items():
            if fid_str.startswith("_"):
                continue
            if int(fid_str) < boundary:
                continue
            for p in people:
                if p.get("id") == tid and p.get("consolidated_id") == old_cid:
                    p["consolidated_id"] = new_cid
        seeds[new_cid] = changed[0][2]
        splits += 1
        logger.info("face_verify: SPLIT tracker %s at frame %s (%s -> %s) new cid=%s",
                    tid, boundary, first_name, changed[0][2], new_cid)
    return splits, seeds


def _split_false_merges(classified, results, raw_faces=None):
    """
    Split a consolidated id merged purely by appearance (identical uniforms
    make every body similarity ~equal, so the engine clusters DIFFERENT people
    into one identity). For each cid holding >=2 trackers, group the trackers
    by their weak face-vote winner; if they fall into different groups the
    engine merged different people, so each later group is minted a fresh
    identity. Two trackers that overlap in time with different face identities
    MUST be different people, so co-occurrence is not a reason to hold back a
    split.

    Additionally, when exactly ONE tracker group has a decisive name but other
    trackers' faces were inconclusive (e.g. the second person is not in the
    DB), split those trackers off too IF their faces clearly disagree with the
    winner's faces. This fixes the "one-sided" false merge: a registered
    person merged with an unregistered one (their DB entry was deleted) would
    otherwise keep the wrong identity, or stay merged forever.
    Returns (number of splits, {new_cid: winner_name}).
    """
    tid_votes = {}
    for fid, boxes in sorted(classified.items()):
        for tid, (name, score) in boxes.items():
            if name is None or score < CONFIDENT_FACE - 0.05:
                continue
            tv = tid_votes.setdefault(tid, {})
            tv[name] = tv.get(name, 0) + 1

    def _winner(votes):
        if not votes:
            return None
        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        if ranked[0][1] < 2:
            return None
        runner = ranked[1][1] if len(ranked) > 1 else 0
        return ranked[0][0] if (ranked[0][1] - runner) >= 1 else None

    # tracker -> (first_frame, last_frame, cid)
    tid_info = {}
    for fid_str, people in results.items():
        if fid_str.startswith("_"):
            continue
        fid = int(fid_str)
        for p in people:
            t = p.get("id")
            c = p.get("consolidated_id")
            if t is None or c is None or c == -1:
                continue
            a, b, _ = tid_info.get(t, (fid, fid, c))
            tid_info[t] = (min(a, fid), max(b, fid), c)

    # tracker -> list of raw face embeddings (for the one-sided check)
    tid_faces = {}
    if raw_faces:
        for fid, boxes in sorted(raw_faces.items()):
            for tid, emb in boxes.items():
                tid_faces.setdefault(tid, []).append(np.asarray(emb, dtype=np.float32))

    def _max_cross_sim(tid_a, tid_b):
        """Best pairwise face similarity between two trackers' raw faces."""
        fa, fb = tid_faces.get(tid_a) or [], tid_faces.get(tid_b) or []
        if not fa or not fb:
            return None
        return max(_face_sim(a, b) for a in fa for b in fb)

    def _mint_split(cid, gtids):
        new_cid = _new_cid(results)
        for g in gtids:
            for fid_str, people in results.items():
                if fid_str.startswith("_"):
                    continue
                for p in people:
                    if p.get("id") == g and p.get("consolidated_id") == cid:
                        p["consolidated_id"] = new_cid
        return new_cid

    cid_tids = {}
    for t, (a, b, c) in tid_info.items():
        cid_tids.setdefault(c, []).append(t)

    splits = 0
    seeds = {}
    for cid, tids in sorted(cid_tids.items(), key=lambda kv: kv[0]):
        tids = sorted(tids, key=int)
        if len(tids) < 2:
            continue
        groups, no_winner = {}, []
        for t in tids:
            w = _winner(tid_votes.get(t, {}))
            (groups.setdefault(w, []).append(t) if w is not None
             else no_winner.append(t))
        if len(groups) >= 2:
            # Classic false merge: multiple trackers each decided a DIFFERENT
            # person. Split every later group off (earliest keeps the cid).
            ordered = sorted(groups.items(),
                             key=lambda kv: min(tid_info[t][0] for t in kv[1]))
            for winner, gtids in ordered[1:]:
                new_cid = _mint_split(cid, gtids)
                seeds[new_cid] = winner
                splits += 1
                logger.info("face_verify: SPLIT false merge cid=%s trackers=%s "
                            "(%s) -> new cid=%s", cid, gtids, winner, new_cid)
            continue
        if len(groups) == 1 and no_winner and tid_faces:
            # One decisive identity + inconclusive trackers. Split off any
            # inconclusive tracker whose faces clearly disagree with the
            # winner's faces (different person whose DB entry is gone).
            winner, wtids = next(iter(groups.items()))
            for t in list(no_winner):
                cross = _max_cross_sim(t, wtids[0])
                if cross is None:
                    continue
                if cross >= _SPLIT_DIFFERENT_FACE:
                    continue  # same person, weak evidence -> keep merged
                new_cid = _mint_split(cid, [t])
                no_winner.remove(t)
                seeds[new_cid] = None  # stays unknown
                splits += 1
                logger.info("face_verify: SPLIT one-sided merge cid=%s "
                            "tracker=%s (face disagrees with %s, cross %.2f) "
                            "-> new cid=%s", cid, t, winner, cross, new_cid)
    return splits, seeds


def _new_cid(results):
    existing = set()
    for fid_str, people in results.items():
        if fid_str.startswith("_"):
            continue
        for p in people:
            if p.get("consolidated_id") is not None:
                existing.add(p["consolidated_id"])
    cid = 1
    while cid in existing:
        cid += 1
    return cid


def _renumber(results):
    """Renumber consolidated ids 1..N by first appearance (skip -1/None).
    Returns the old->new mapping."""
    first_seen = {}
    for fid_str in sorted((k for k in results if not k.startswith("_")), key=int):
        fid = int(fid_str)
        for p in results[fid_str]:
            c = p.get("consolidated_id")
            if c is None or c == -1:
                continue
            first_seen.setdefault(c, fid)
    remap = {c: i + 1 for i, c in enumerate(
        sorted(first_seen, key=lambda c: first_seen[c]))}
    for fid_str, people in results.items():
        if fid_str.startswith("_"):
            continue
        for p in people:
            c = p.get("consolidated_id")
            if c is not None and c != -1:
                p["consolidated_id"] = remap[c]
    return remap


def verify_and_fix(video_path, tracking_json_path, reid_json_path, identity_db,
                   stride=4, fps=25.0, max_faces=60):
    """
    Run the face-first verification pass on an existing reid.json and rewrite
    it in place. Returns a people summary in the same shape as
    run_camera_reid, or None if no faces could be classified.

    Conservative by design: names/ids are only changed when the face votes
    are decisive; otherwise the engine's output is preserved.
    """
    with open(reid_json_path, encoding="utf-8") as f:
        results = json.load(f)
    old_tracks = results.pop("__tracks__", {}) or {}
    cached_faces = results.pop("__verify_faces__", {}) or {}

    frame_ids = sorted(int(k) for k in results if not k.startswith("_"))

    from reidentification.insight_face import get_shared_extractor
    extractor = get_shared_extractor()
    if not extractor.enabled:
        logger.warning("face_verify: no InsightFace available — pass skipped")
        results["__tracks__"] = old_tracks
        return None

    if cached_faces:
        # The engine already extracted (frame, embedding) per tracker while it
        # had the video decoded — reuse that cache instead of decoding the
        # video again. Only keep frames that are actually in the reid output.
        # NOTE: tracker ids are kept as STRINGS — results store p["id"] as a
        # string (DeepSORT emits string ids), and every lookup in this module
        # compares against p["id"], so int()-casting here would make every
        # lookup silently miss (zero swaps/splits/votes).
        frame_set = set(frame_ids)
        raw_faces = {}
        for tid, arr in cached_faces.items():
            for fid, emb in arr:
                fid = int(fid)
                if fid not in frame_set:
                    continue
                raw_faces.setdefault(fid, {})[tid] = np.asarray(emb, dtype=np.float32)
    else:
        # Fallback (standalone verification of an older reid.json): decode the
        # video and re-extract faces at `stride`.
        with open(tracking_json_path, encoding="utf-8") as f:
            tracking_data = json.load(f)
        raw_faces = _collect_faces(video_path, tracking_data, frame_ids, stride, extractor)

    if not raw_faces:
        logger.warning("face_verify: no faces found — leaving output unchanged")
        results["__tracks__"] = old_tracks
        return None

    classifier = _FaceClassifier(identity_db)
    classified = {
        fid: {tid: classifier.classify(face) for tid, face in boxes.items()}
        for fid, boxes in raw_faces.items()
    }

    n_swap = _detect_and_fix_swaps(classified, results)
    n_split, seeds = _split_person_switches(classified, results)
    n_false, seeds2 = _split_false_merges(classified, results, raw_faces)
    seeds.update(seeds2)

    # Face votes per consolidated id (pre-renumber, after corrections).
    votes, max_score, face_best = {}, {}, {}
    for fid, boxes in sorted(classified.items()):
        for tid, (name, score) in boxes.items():
            cid = _cid_at(results, fid, tid)
            if cid is None:
                continue
            if name is not None and score >= CONFIDENT_FACE:
                v = votes.setdefault(cid, {})
                v[name] = v.get(name, 0) + 1
            max_score[cid] = max(max_score.get(cid, 0.0), score)
            if name is not None:
                fb = face_best.setdefault(cid, {})
                fb[name] = max(fb.get(name, 0.0), score)

    def _face_decided(cid):
        """(name, similarity, face_sim, cues) from decisive face votes, else None.
        Thresholds are adaptive: a single near-perfect face is decisive, weaker
        evidence needs more votes and a margin."""
        v = votes.get(cid, {})
        if not v:
            return None
        ranked = sorted(v.items(), key=lambda kv: kv[1], reverse=True)
        top, top_v = ranked[0]
        runner_v = ranked[1][1] if len(ranked) > 1 else 0
        ms = max_score.get(cid, 0.0)
        if ms >= 0.90:
            min_v, margin = 1, 0
        elif ms >= 0.70:
            min_v, margin = 2, 1
        else:
            min_v, margin = MIN_VOTES, MARGIN_VOTES
        if top_v >= min_v and (top_v - runner_v) >= margin and ms >= CONFIDENT_FACE:
            fs = face_best.get(cid, {}).get(top)
            return (top, round(float(ms), 3),
                    round(float(fs), 3) if fs is not None else None, ["face"])
        return None

    # Per-cid decision BEFORE renumber so old_tracks keys still align.
    cid_set = set()
    for fid_str, people in results.items():
        if fid_str.startswith("_"):
            continue
        for p in people:
            c = p.get("consolidated_id")
            if c is not None and c != -1:
                cid_set.add(c)

    decision = {}
    for cid in sorted(cid_set):
        old = old_tracks.get(str(cid), {})
        d = _face_decided(cid)
        if d is not None:
            decision[cid] = d
        elif cid in seeds:
            # split-created identity: name comes from the splitter's face winner
            decision[cid] = (seeds[cid], None, None, ["face"])
        elif old.get("name") and not old.get("cues") \
                and old.get("similarity") is None:
            # previously user-corrected in the UI (no engine metrics)
            decision[cid] = (old.get("name"), old.get("similarity"),
                             old.get("face_sim"), old.get("cues", []))
        elif old.get("name") and (old.get("face_sim") or 0) >= CONFIDENT_FACE:
            # engine face-named with a strong face score
            decision[cid] = (old.get("name"), old.get("similarity"),
                             old.get("face_sim"), old.get("cues", []))
        else:
            # appearance-only name (unreliable in identical uniforms) -> unknown
            decision[cid] = (None, None, None, [])

    remap = _renumber(results)
    decision_new = {remap[c]: v for c, v in decision.items() if c in remap}

    # Rebuild per-frame face pool per consolidated id (post-correction).
    cid_faces = {}
    for fid, boxes in sorted(raw_faces.items()):
        for tid, face in boxes.items():
            cid = _cid_at(results, fid, tid)
            if cid is None:
                continue
            cid_faces.setdefault(cid, []).append(face)

    tracks_payload = {}
    for cid in remap.values():
        name, sim, fsim, cues = decision_new.get(cid, (None, None, None, []))
        old = old_tracks.get(str(cid))
        faces = [f.tolist() for f in cid_faces.get(cid, [])[:max_faces]]
        tracks_payload[str(cid)] = {
            "original_sid": (old or {}).get("original_sid", cid),
            "faces": faces,
            "mean_feature": (old or {}).get("mean_feature"),
            "name": name,
            "similarity": sim,
            "face_sim": fsim,
            "cues": cues,
        }

    results["__tracks__"] = tracks_payload
    with open(reid_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    first_seen = {}
    last_seen = {}
    for fid_str in sorted((k for k in results if not k.startswith("_")), key=int):
        fid = int(fid_str)
        for p in results[fid_str]:
            c = p.get("consolidated_id")
            if c is None or c == -1:
                continue
            first_seen.setdefault(c, fid)
            last_seen[c] = fid

    people = []
    for cid in sorted(first_seen):
        name, sim, fsim, cues = decision_new.get(cid, (None, None, None, []))
        people.append({
            "track_id": cid,
            "name": name,
            "similarity": sim,
            "face_sim": fsim,
            "cues": cues,
            "first_seen_sec": round(first_seen[cid] / fps, 1),
            "last_seen_sec": round(last_seen[cid] / fps, 1),
        })

    named = sum(1 for v in decision_new.values() if v[0] is not None)
    logger.info("face_verify: %s swap(s), %s split(s), %s false-merge split(s); "
                "%s/%s identities named", n_swap, n_split, n_false, named,
                len(tracks_payload))
    return people
