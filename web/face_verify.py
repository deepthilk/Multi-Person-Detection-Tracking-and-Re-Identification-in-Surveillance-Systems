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

from registration.db_config import SEARCH_SETTINGS as _DB_SEARCH_SETTINGS

logger = logging.getLogger(__name__)

# InsightFace ArcFace scale measured on this project's footage:
# same person >= ~0.54, different people <= ~0.28.
CONFIDENT_FACE = 0.50      # a face this strong is trusted to name a box
# A face at/above the DB confirmed threshold can witness a tracker SWITCH
# (DeepSORT hands the same id to a different person; the old 0.58 bar missed
# low-res faces like Usha's 0.44-0.49, leaving a contaminated tracker merged).
SPLIT_MIN_FACE = _DB_SEARCH_SETTINGS["face_confirmed_threshold"]
SWAP_SCORE_MIN = 0.50      # reciprocal switch needs both sides this confident
MIN_VOTES = 3              # fewest confident faces before we trust a name
MARGIN_VOTES = 2           # top person must beat the runner-up by this many votes
# A face at/above the DB's face_confirmed_threshold is what match_multimodal
# already treated as a confirmed same-person match; keep the engine's
# face-named result in that case instead of re-wiping it with the stricter
# CONFIDENT_FACE bar (real case: Prajna's low-res registration photos score
# 0.44-0.48 vs 0.50 here, so a correctly engine-named track became "unknown").
_DB_FACE_CONFIRMED = _DB_SEARCH_SETTINGS["face_confirmed_threshold"]
# Max gap (in frames) between a tracker's segment switch and the reciprocal
# tracker's switch before they count as the same swap event.
SWAP_WINDOW = 30
# A tracker's faces this far below another tracker's faces are treated as a
# DIFFERENT person for the one-sided false-merge split (same-person >= ~0.54,
# different-person <= ~0.28 on this footage, so 0.40 is a safe midpoint).
_SPLIT_DIFFERENT_FACE = 0.40
# Reappearance-fragment floor: when two groups in one cid are TIME-DISJOINT
# (a person left and returned — DeepSORT hands them new tracker ids) and the
# split-off group's face votes are RELATIVE-ONLY (weak, no confident winner),
# only a CLEAR face mismatch is reason to split. Relative votes on low-res
# unregistered faces are noise (real case: Lekha's cam2 segments voted
# 'pranjali' / 'Prajna' on galleries she doesn't belong to, cross-face 0.39)
# and splitting re-mints the SAME person a second id. Same-person fragments
# here cross at >= 0.39 (deeps 0.82, Prajna 0.54, Lekha 0.39); genuinely
# different disjoint people sit well below 0.35.
_REAPPEAR_DIFFERENT_FACE = 0.35
# Face-to-face reappearance merge (_merge_fragments): when the same person
# leaves and comes back, DeepSORT hands them a NEW tracker id AND the face-vote
# NAME can fail to confirm the reappearance (the new segment's faces can be
# weak and even vote for the WRONG low-res gallery — Prajna's reappearance on
# cam2 scores 0.25-0.38 and votes 'deeps'). Two time-disjoint fragments are
# still the SAME person when enough of their CROSS-FACE pairs are close, no
# gallery involved. Measured on this footage: same-person tops 0.57-0.76 with
# many pairs >= 0.5 (cam1 reappearance 26/165, cam2 reappearance 1570/3721);
# different people top <= 0.48 with ZERO pairs >= 0.5 — 0.55 / 3 pairs sits
# safely between.
FACE_MERGE_TOP = 0.55     # best pairwise sim required to even consider a merge
FACE_MERGE_SIM = 0.50     # a pair at/above this counts as a same-person pair
FACE_MERGE_PAIRS = 3      # fewest same-person pairs before trusting the merge


def _face_sim(a, b) -> float:
    from reidentification.insight_face import InsightFaceExtractor
    return float(InsightFaceExtractor.similarity(
        np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))


def _box_iou(a, b) -> float:
    """Intersection-over-union of two [x0, y0, x1, y1] boxes (>= 0 on miss)."""
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = (float(v) for v in a[:4])
    bx0, by0, bx1, by1 = (float(v) for v in b[:4])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    aa = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    bb = max(0.0, (bx1 - bx0) * (by1 - by0))
    uni = aa + bb - inter
    return inter / uni if uni > 0 else 0.0


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


def _split_person_switches(classified, results, engine_names=None):
    """
    Split a tracker whose faces show a clean person change with no reciprocal
    counterpart (a DeepSORT id that stopped following one person and picked up
    another). The segment that DIFFERS from the cid's engine-assigned name is
    minted a fresh identity:

      • engine named the cid after the FIRST person  -> mint the later segment
        (a different person the tracker picked up);
      • engine named the cid after the LATER person  -> mint the early segment
        (the different person was on screen first, e.g. Usha entering before
        deeps, then DeepSORT handing the SAME id to deeps when she left).

    Returns the number of splits.
    """
    stretches = {}
    for fid, boxes in sorted(classified.items()):
        for tid, (name, score) in boxes.items():
            if name is None or score < SPLIT_MIN_FACE:
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
        # Noise guard: a single stray face of another person must not split.
        if changed[0][1] - changed[0][0] < 1:
            continue
        boundary = changed[0][0]
        old_cid = _cid_at(results, min(r[0] for r in runs), tid)
        if old_cid is None:
            continue
        new_cid = _new_cid(results)
        old_name = engine_names.get(old_cid) if engine_names else None
        mint_early = old_name is not None and old_name == changed[0][2]
        if mint_early:
            # cid belongs to the LATER person -> move the EARLY segment out.
            for fid_str, people in results.items():
                if fid_str.startswith("_"):
                    continue
                if int(fid_str) >= boundary:
                    continue
                for p in people:
                    if p.get("id") == tid and p.get("consolidated_id") == old_cid:
                        p["consolidated_id"] = new_cid
            seeds[new_cid] = first_name
            logger.info("face_verify: SPLIT tracker %s early at frame %s "
                        "(%s -> %s, cid belongs to %s) new cid=%s",
                        tid, boundary, first_name, changed[0][2], old_name, new_cid)
        else:
            # cid belongs to the FIRST person -> move the LATER segment out.
            for fid_str, people in results.items():
                if fid_str.startswith("_"):
                    continue
                if int(fid_str) < boundary:
                    continue
                for p in people:
                    if p.get("id") == tid and p.get("consolidated_id") == old_cid:
                        p["consolidated_id"] = new_cid
            seeds[new_cid] = changed[0][2]
            logger.info("face_verify: SPLIT tracker %s later at frame %s "
                        "(%s -> %s, cid belongs to %s) new cid=%s",
                        tid, boundary, first_name, changed[0][2], old_name, new_cid)
        splits += 1
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
    # (cid, tid) -> face-vote tally, from per-frame cid attribution.
    # A single tracker can span several cids after _split_person_switches
    # minted an early/late segment, so votes must follow the cid the tracker
    # actually has AT THAT FRAME (same rule as the final naming votes).
    tid_votes = {}
    for fid, boxes in sorted(classified.items()):
        for tid, (name, score) in boxes.items():
            if name is None or score < CONFIDENT_FACE - 0.05:
                continue
            cid = _cid_at(results, fid, tid)
            if cid is None:
                continue
            tv = tid_votes.setdefault((cid, tid), {})
            tv[name] = tv.get(name, 0) + 1

    # RELATIVE tally with no absolute score gate: every face contributes its
    # best-match name. Confident-vote tallies are empty when faces are small /
    # low-res (Usha scores 0.23-0.42), so a cid of weak-faced trackers can have
    # NO confident winner and the false-merge splitter would skip them even
    # though each tracker still has a consistent RELATIVE preference (tid2 ->
    # deeps 3/3, tid4 -> usha 6/9). Used only when no tracker is confident.
    rel_votes = {}
    for fid, boxes in sorted(classified.items()):
        for tid, (name, score) in boxes.items():
            if name is None:
                continue
            cid = _cid_at(results, fid, tid)
            if cid is None:
                continue
            rv = rel_votes.setdefault((cid, tid), {})
            rv[name] = rv.get(name, 0) + 1

    def _winner(votes):
        if not votes:
            return None
        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        if ranked[0][1] < 2:
            return None
        runner = ranked[1][1] if len(ranked) > 1 else 0
        return ranked[0][0] if (ranked[0][1] - runner) >= 1 else None

    # (cid, tid) -> (first_frame, last_frame)
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
            a, b = tid_info.get((c, t), (fid, fid))
            tid_info[(c, t)] = (min(a, fid), max(b, fid))

    # (cid, tid) -> list of raw face embeddings (for the one-sided check)
    tid_faces = {}
    if raw_faces:
        for fid, boxes in sorted(raw_faces.items()):
            for tid, emb in boxes.items():
                cid = _cid_at(results, fid, tid)
                if cid is None:
                    continue
                tid_faces.setdefault((cid, tid), []).append(np.asarray(emb, dtype=np.float32))

    def _max_cross_sim(key_a, key_b):
        """Best pairwise face similarity between two (cid, tid) face sets."""
        fa, fb = tid_faces.get(key_a) or [], tid_faces.get(key_b) or []
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

    def _spans_disjoint(cid, ga, gb):
        """Two tracker groups never share a frame (same person can't be in two
        places). Reappearance fragments of ONE person are always disjoint; two
        different people can co-occur, so disjoint spans are a strong signal
        the engine merged a genuine reappearance, not two distinct people."""
        def _range(gtids):
            spans = [tid_info[(cid, t)] for t in gtids if (cid, t) in tid_info]
            if not spans:
                return None
            return (min(s[0] for s in spans), max(s[1] for s in spans))
        ra, rb = _range(ga), _range(gb)
        if ra is None or rb is None:
            return False
        return ra[1] < rb[0] or rb[1] < ra[0]

    def _preserve_reappearance(cid, gtids, kept_tids):
        """True when the split-off group is a weak reappearance fragment of the
        kept group: time-disjoint spans (the person left and returned), a
        RELATIVE-only face winner (no confident votes — unregistered/low-res
        faces), and no clear cross-face disagreement. Splitting such a group
        re-mints the SAME person a second id (real case: Lekha cam2)."""
        if not all(t in rel_override for t in gtids):
            return False
        if not _spans_disjoint(cid, gtids, kept_tids):
            return False
        cross = _max_cross_sim((cid, gtids[0]), (cid, kept_tids[0]))
        if cross is None or cross < _REAPPEAR_DIFFERENT_FACE:
            return False
        logger.info("face_verify: PRESERVE reappearance merge cid=%s "
                    "trackers=%s (weak relative vote, cross %.2f)",
                    cid, gtids, cross)
        return True

    cid_tids = {}
    for (c, t), (a, b) in tid_info.items():
        cid_tids.setdefault(c, []).append(t)

    splits = 0
    seeds = {}
    for cid, tids in sorted(cid_tids.items(), key=lambda kv: kv[0]):
        tids = sorted(tids, key=int)
        if len(tids) < 2:
            continue
        groups, no_winner, rel_override = {}, [], set()
        for t in tids:
            w = _winner(tid_votes.get((cid, t), {}))
            rw = _winner(rel_votes.get((cid, t), {}))
            if rw is not None and (w is None or rw != w):
                # The confident tally can be a few high-scoring strays while
                # the overwhelming RELATIVE majority says someone else (real
                # case: the 3rd/unregistered person on cam1 — 4 faces >= 0.45
                # hit deeps at 0.576, but 77 faces prefer pranjali). Prefer the
                # relative winner when it has a strong majority — also when the
                # confident tally has no winner at all (a single stray face is
                # not enough to name a tracker, but 25/33 faces agreeing on one
                # gallery is a real identity signal).
                rv = rel_votes.get((cid, t), {})
                top = rv.get(rw, 0)
                runner = sorted(rv.values(), reverse=True)[1] if len(rv) > 1 else 0
                if top >= MIN_VOTES and (top - runner) >= MARGIN_VOTES:
                    w = rw
                    rel_override.add(t)
            (groups.setdefault(w, []).append(t) if w is not None
             else no_winner.append(t))
        if len(groups) >= 2:
            # Classic false merge: multiple trackers each decided a DIFFERENT
            # person. Split every later group off (earliest keeps the cid).
            ordered = sorted(groups.items(),
                             key=lambda kv: min(tid_info[(cid, t)][0] for t in kv[1]))
            for winner, gtids in ordered[1:]:
                if _preserve_reappearance(cid, gtids, ordered[0][1]):
                    continue
                new_cid = _mint_split(cid, gtids)
                if all(t in rel_override for t in gtids):
                    # Name came only from the relative tally — the person is
                    # NOT confidently any registered face (unregistered 3rd
                    # person), so keep the split identity UNIDENTIFIED.
                    seeds[new_cid] = None
                else:
                    seeds[new_cid] = winner
                splits += 1
                logger.info("face_verify: SPLIT false merge cid=%s trackers=%s "
                            "(%s) -> new cid=%s", cid, gtids, winner, new_cid)
            continue
        if len(groups) == 0 and len(no_winner) >= 2:
            # ZERO trackers have a confident face name — faces here are all
            # small/low-res (0.23-0.42 < vote bar). But each tracker still has a
            # consistent RELATIVE preference (best-match name per face, no gate).
            # Two trackers preferring DIFFERENT people are a false merge the
            # engine created by identical-uniform appearance; split them. The
            # engine's own appearance-only verdict (e.g. face=0.457) is NOT
            # trusted here because identical uniforms make everyone's faces look
            # mutually similar (each tracker's within-sim can be BELOW its
            # cross-sim to another person).
            rel_groups, rel_no = {}, []
            for t in no_winner:
                w = _winner(rel_votes.get((cid, t), {}))
                (rel_groups.setdefault(w, []).append(t) if w is not None
                 else rel_no.append(t))
            if len(rel_groups) >= 2:
                ordered = sorted(rel_groups.items(),
                                 key=lambda kv: min(tid_info[(cid, t)][0] for t in kv[1]))
                for winner, gtids in ordered[1:]:
                    if _preserve_reappearance(cid, gtids, ordered[0][1]):
                        continue
                    new_cid = _mint_split(cid, gtids)
                    seeds[new_cid] = winner
                    splits += 1
                    logger.info("face_verify: SPLIT zero-confident false merge "
                                "cid=%s trackers=%s (rel. %s) -> new cid=%s",
                                cid, gtids, winner, new_cid)
                continue
            if len(rel_groups) == 1 and rel_no and tid_faces:
                # One relative winner + stragglers whose own faces disagreed
                # with everyone else (their relative tally flipped too often).
                winner, wtids = next(iter(rel_groups.items()))
                for t in list(rel_no):
                    cross = _max_cross_sim((cid, t), (cid, wtids[0]))
                    if cross is None:
                        continue
                    if cross >= _SPLIT_DIFFERENT_FACE:
                        continue  # same person, weak evidence -> keep merged
                    new_cid = _mint_split(cid, [t])
                    rel_no.remove(t)
                    seeds[new_cid] = None  # stays unknown
                    splits += 1
                    logger.info("face_verify: SPLIT zero-confident one-sided "
                                "merge cid=%s tracker=%s (face disagrees with "
                                "%s, cross %.2f) -> new cid=%s",
                                cid, t, winner, cross, new_cid)
            continue
        if len(groups) == 1 and no_winner and tid_faces:
            # One decisive identity + inconclusive trackers. Split off any
            # inconclusive tracker whose faces clearly disagree with the
            # winner's faces (different person whose DB entry is gone).
            winner, wtids = next(iter(groups.items()))
            for t in list(no_winner):
                cross = _max_cross_sim((cid, t), (cid, wtids[0]))
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

    # ── Co-occurrence safety net ─────────────────────────────────────────────
    # After the face-based passes, NO identity may ever hold two different
    # trackers in the SAME frame — one person can't be in two places at once.
    # The engine occasionally merges two co-occurring people into one cid
    # (identical uniforms make their appearances ~equal), and if their faces
    # are inconclusive every vote-based split above correctly declines. That
    # leaves BOTH boxes labelled with the same global id on one frame (user
    # visible: "one frame has two people with the same GID"). Two distinct
    # DeepSORT track ids in the same frame are by definition different people,
    # so this is a guaranteed false merge — split the later-appearing ones off
    # (earliest tracker keeps the cid; split identities stay unidentified).
    frame_tids = {}
    for fid_str, people in results.items():
        if fid_str.startswith("_"):
            continue
        for p in people:
            c = p.get("consolidated_id")
            t = p.get("id")
            if c is None or c == -1 or t is None:
                continue
            frame_tids.setdefault((int(fid_str), c), set()).add(t)

    first_by_cid = {}
    for fid_str, people in results.items():
        if fid_str.startswith("_"):
            continue
        for p in people:
            c = p.get("consolidated_id")
            if c is None or c == -1:
                continue
            first_by_cid.setdefault(c, int(fid_str))

    cooc_cids = sorted({c for (_, c), tids in frame_tids.items() if len(tids) >= 2})
    for cid in cooc_cids:
        # trackers that ever share a frame with another tracker of the same cid
        conflicting = set()
        for (fid, c), tids in sorted(frame_tids.items()):
            if c != cid or len(tids) < 2:
                continue
            conflicting.update(tids)
        if len(conflicting) < 2:
            continue
        keep = min(conflicting, key=lambda t: (tid_info.get((cid, t), (10**9, 0))[0], t))
        rest = sorted(conflicting - {keep}, key=lambda t: (tid_info.get((cid, t), (10**9, 0))[0], t))
        for t in rest:
            new_cid = _mint_split(cid, [t])
            seeds[new_cid] = None  # distinct person, but no confident name
            splits += 1
            logger.info("face_verify: SPLIT same-frame merge cid=%s tracker=%s "
                        "(co-occurs with %s in the same frame) -> new cid=%s",
                        cid, t, keep, new_cid)
    return splits, seeds


def _merge_fragments(classified, results, seeds=None, raw_faces=None):
    """
    Re-merge identities the engine FRAGMENTED. When the same person leaves and
    returns, DeepSORT often hands them a NEW tracker id, and the engine's
    appearance-only reappearance pass can miss the match (appearance is weak in
    uniform-heavy footage) — minting a second identity for the SAME person.
    Fragments also arise from this module's own splits: once a false merge is
    split or a tracker is split mid-person, each segment carries only PART of
    the person's faces, so a fragment that never reaches a decisive face name
    still belongs with its siblings (real case: Usha's faces score 0.44-0.49,
    below CONFIDENT_FACE, so her fragments never became "decisive" and stayed
    forever apart).

    Two gates, either of which merges two time-disjoint fragments:

    (1) FACE-VOTE NAME — the decisive name computed exactly as the final naming
    does (see _face_decided), then the SPLITTER'S SEED name (a split-created
    identity's name is the splitter's face winner), then the RELATIVE best-match
    winner. The relative winner now needs an ABSOLUTE score floor: a fragment
    whose best face scores below the DB-confirmed bar can vote for the WRONG
    gallery (Prajna's low-res reappearance faces score 0.25-0.38 and vote
    'deeps'), so only faces the DB itself confirms may carry a relative name.
    Two identities merge when they agree on a name AND at least one side is
    STRONG (decisive name or splitter seed):

      • genuine fragments merge — e.g. two "usha" fragments where Usha's faces
        only ever reach the relative bar (0.44-0.49 >= the floor), or a
        relative "deeps" fragment folded into a decisive "deeps" fragment;
      • different people NEVER merge on weak evidence alone — a relative-only
        vote (both sides weak) is not enough, so two people who merely look
        like the same low-res gallery blob stay separate;
      • the engine's own name is NEVER trusted: identical uniforms make body
        appearance ~equal, so the engine routinely merges DIFFERENT people into
        one identity (the splitter then un-merges them), and a split-created
        identity has no engine name at all.

    (2) FACE-TO-FACE — the gallery-independent fallback for reappearances whose
    names never agree (a weak reappearance can vote for a DIFFERENT low-res
    gallery than its first segment). Two fragments are the SAME person when
    enough of their raw CROSS-FACE pairs are close (see FACE_MERGE_*): time
    was already proven disjoint, and cross-face similarity does not depend on
    registration photo quality at all.

    The later fragment is folded into the earliest. Returns the number of
    merges performed.
    """
    votes, max_score = {}, {}
    rel_votes = {}
    for fid, boxes in sorted(classified.items()):
        for tid, (name, score) in boxes.items():
            cid = _cid_at(results, fid, tid)
            if cid is None:
                continue
            if name is not None and score >= CONFIDENT_FACE:
                v = votes.setdefault(cid, {})
                v[name] = v.get(name, 0) + 1
            if name is not None:
                rv = rel_votes.setdefault(cid, {})
                rv[name] = rv.get(name, 0) + 1
            max_score[cid] = max(max_score.get(cid, 0.0), score)

    def _decided(cid):
        v = votes.get(cid, {})
        if not v:
            return None
        ranked = sorted(v.items(), key=lambda kv: kv[1], reverse=True)
        top, top_v = ranked[0]
        runner_v = ranked[1][1] if len(ranked) > 1 else 0
        ms = max_score.get(cid, 0.0)
        if ms >= 0.90:
            min_v, margin = 2, 0
        elif ms >= 0.70:
            min_v, margin = 2, 1
        else:
            min_v, margin = MIN_VOTES, MARGIN_VOTES
        if top_v >= min_v and (top_v - runner_v) >= margin and ms >= CONFIDENT_FACE:
            return top
        return None

    def _rel_winner(cid):
        """Best-match-name majority, gated on an ABSOLUTE score floor (the DB's
        own face-confirmed threshold). A relative tally is the same weak-face
        evidence as the false-merge splitter uses, but an UNGATED one is
        unsafe for NAMING: a fragment whose best face sits far below the
        confirmed bar can vote for the WRONG gallery (Prajna's reappearance on
        cam2 scores 0.25-0.38 and votes 'deeps' because the deeps gallery is
        higher-res). Only faces the DB itself would confirm (>= the floor) may
        carry a relative name — Usha's 0.44-0.49 still passes."""
        if max_score.get(cid, 0.0) < _DB_FACE_CONFIRMED:
            return None
        v = rel_votes.get(cid, {})
        if not v:
            return None
        ranked = sorted(v.items(), key=lambda kv: kv[1], reverse=True)
        if ranked[0][1] < 2:
            return None
        runner = ranked[1][1] if len(ranked) > 1 else 0
        return ranked[0][0] if (ranked[0][1] - runner) >= 1 else None

    def _name(cid):
        """(name, strong) — decisive name or splitter seed is strong; a bare
        relative winner is weak and can only merge INTO a strong fragment."""
        s = (seeds or {}).get(cid)
        if cid in (seeds or {}) and s is None:
            # Split-created UNIDENTIFIED identity (an unregistered person the
            # splitter explicitly set apart). A few stray face hits >= 0.50
            # must NOT rename it and fold it back into the person it was
            # separated from — the relative tally says the faces belong to a
            # DIFFERENT person. A None-seeded identity never merges.
            return None, True
        d = _decided(cid)
        if d is not None:
            return d, True
        s = (seeds or {}).get(cid)
        if s is not None:
            return s, True
        w = _rel_winner(cid)
        if w is None:
            return None, False
        # A relative winner with a STRONG majority counts as strong: Usha's
        # faces never reach CONFIDENT_FACE (0.44-0.49), yet every one of her
        # 19 faces votes 'usha' — so two such fragments are the SAME person
        # even though neither alone names her decisively. A fragment with a
        # bare 2-vote plurality stays weak (two people in identical uniforms
        # can both look like the same low-res gallery blob).
        rv = rel_votes.get(cid, {})
        top = rv.get(w, 0)
        runner = sorted(rv.values(), reverse=True)[1] if len(rv) > 1 else 0
        strong_rel = top >= MIN_VOTES and (top - runner) >= MARGIN_VOTES
        return w, strong_rel

    # Fragments of the SAME person can never be on screen at the same time.
    # Two identities whose tracker spans overlap are DIFFERENT people (real
    # case: a crowded clip where several people each weakly vote for the same
    # gallery blob — merging them mints one fake identity). Disjoint spans are
    # a hard prerequisite for merging.
    cid_frames = {}
    for fid_str, people in results.items():
        if fid_str.startswith("_"):
            continue
        for p in people:
            c = p.get("consolidated_id")
            if c is None or c == -1:
                continue
            cid_frames.setdefault(c, set()).add(int(fid_str))

    # Raw cross-face pools per consolidated id (post-correction) for the
    # gallery-independent face-to-face merge gate, plus the frames each cid
    # actually has face evidence on (the correct time gate for that gate).
    cid_pool = {}
    cid_face_frames = {}
    if raw_faces:
        for fid, boxes in sorted(raw_faces.items()):
            for tid, emb in boxes.items():
                cid = _cid_at(results, fid, tid)
                if cid is None:
                    continue
                cid_pool.setdefault(cid, []).append(np.asarray(emb, dtype=np.float32))
                cid_face_frames.setdefault(cid, set()).add(fid)

    cids = sorted(set(votes) | set(rel_votes))
    if len(cids) < 2:
        return 0, {}

    parent = {c: c for c in cids}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def _root_pool(c):
        """All raw faces pooled across the whole component of `c` (so a merge
        chain accumulates evidence instead of only the seed pair's faces)."""
        r = find(c)
        out = []
        for cc, arr in cid_pool.items():
            if find(cc) == r:
                out.extend(arr)
        return out

    def _root_face_frames(c):
        """All face-bearing frames across the whole component of `c`."""
        r = find(c)
        out = set()
        for cc, frames in cid_face_frames.items():
            if find(cc) == r:
                out |= frames
        return out

    def _same_person_faces(a, b):
        """Same-person verdict from raw face pairs alone — no gallery, no name.
        Same-person reappearances on this footage top at 0.57-0.76 with many
        pairs >= 0.50; different people top <= 0.48 with zero pairs >= 0.50."""
        fa, fb = _root_pool(a), _root_pool(b)
        if len(fa) < FACE_MERGE_PAIRS or len(fb) < FACE_MERGE_PAIRS:
            return False
        top, n_ge = 0.0, 0
        for x in fa:
            for y in fb:
                s = _face_sim(x, y)
                if s > top:
                    top = s
                if s >= FACE_MERGE_SIM:
                    n_ge += 1
                    if top >= FACE_MERGE_TOP and n_ge >= FACE_MERGE_PAIRS:
                        return True
        return top >= FACE_MERGE_TOP and n_ge >= FACE_MERGE_PAIRS

    def _face_spans_disjoint(a, b):
        """The FACE-BEARING spans of two components are disjoint. The full-box
        span of a fragment can overlap another's because the engine folds
        unrelated face-less boxes into it (real case: cam1's first Prajna
        segment carries the reappearing chair and a face-less passer-by in its
        id, extending its span past the reappearance's start) — but the face
        evidence, which is what proves a reappearance, never overlaps. The
        same person can never show two faces at once, so disjoint FACE spans
        are the correct time gate for the face-to-face merge."""
        fa, fb = _root_face_frames(a), _root_face_frames(b)
        if not fa or not fb:
            return False
        return fa.isdisjoint(fb)

    merged_pairs = 0
    for i, a in enumerate(cids):
        for b in cids[i + 1:]:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            fa, fb = cid_frames.get(ra, set()), cid_frames.get(rb, set())
            name_time_ok = not (fa and fb and not fa.isdisjoint(fb))
            na, sa = _name(a)
            nb, sb = _name(b)
            why = None
            if name_time_ok and na and nb and na == nb and (sa or sb):
                why = f"both '{na}'"
            elif _face_spans_disjoint(a, b) and _same_person_faces(a, b):
                why = "face"
            if why is None:
                continue            # need a shared face-vote name (time-disjoint
                                    # spans) OR a face-to-face same-person verdict
                                    # (time-disjoint face spans)
            union(a, b)
            cid_frames.setdefault(find(a), set()).update(fa | fb)
            merged_pairs += 1
            logger.info("face_verify: MERGE fragmented ids %s + %s (%s)",
                        a, b, why)

    if not merged_pairs:
        return 0, {}

    root_by_cid = {c: find(c) for c in cids}
    # Fold every non-root fragment into its root; the smallest cid of the
    # component is kept so the renumbering that follows stays stable.
    roots = {}
    for c, r in root_by_cid.items():
        roots.setdefault(r, []).append(c)
    keep = {min(group): r for r, group in roots.items()}

    n_merged = 0
    merged_names = {}
    for c, r in root_by_cid.items():
        target = keep[r]
        if c == target:
            continue
        for fid_str, people in results.items():
            if fid_str.startswith("_"):
                continue
            for p in people:
                if p.get("consolidated_id") == c:
                    p["consolidated_id"] = target
        n_merged += 1
    # The merged root inherits the agreed face-vote name: fragments fold into
    # the earliest cid, and the naming pass runs after this, so record it here
    # (real case: Usha's merged root kept no decisive votes — 0.44-0.49 faces —
    # and the old-name preservation in verify_and_fix also rejected it, wiping
    # her name). seeds are per-cid; rebuild the agreed name for each root.
    for c in roots:
        for g in roots[c]:
            na, _ = _name(g)
            if na:
                merged_names.setdefault(keep[c], na)
    logger.info("face_verify: %s fragmented identities folded into %s root(s)",
                n_merged, len(roots))
    return n_merged, merged_names


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


def _non_person_cids(results, cid_faces, frame_h):
    """Consolidated ids that hold NO face at all and whose bbox never looks
    like a person: FULLY STATIC (an inanimate object — DeepSORT tracks the
    chair at the side, h ~365 of a 720-tall frame, as a person) or SHORT/TINY
    (a partial edge sliver, max height < 60% of the frame). A real person can
    be face-less only if far away, but then they move and are full-height; a
    still real person still yields at least one readable face. Zero-face
    static/tiny boxes are non-people: drop them so they never show up as a
    fake person id (real case: the user's session showed a fake id for a chair
    at the side). Returns the list of cids to suppress."""
    stats = {}
    for fid_str, people in results.items():
        if fid_str.startswith("_"):
            continue
        for p in people:
            c = p.get("consolidated_id")
            bb = p.get("bbox")
            if c is None or c == -1 or not bb:
                continue
            x0, y0, x1, y1 = (float(v) for v in bb[:4])
            s = stats.setdefault(c, {"cx": [], "cy": [], "w": [], "h": []})
            s["cx"].append((x0 + x1) / 2.0)
            s["cy"].append((y0 + y1) / 2.0)
            s["w"].append(x1 - x0)
            s["h"].append(y1 - y0)
    out = []
    for c, s in stats.items():
        if cid_faces.get(c):
            continue
        if len(s["cx"]) < 2:
            continue
        static = (max(s["cx"]) - min(s["cx"]) < 15.0
                  and max(s["cy"]) - min(s["cy"]) < 15.0
                  and max(s["w"]) - min(s["w"]) < 15.0
                  and max(s["h"]) - min(s["h"]) < 15.0)
        short = max(s["h"]) < 0.6 * frame_h
        if static or short:
            out.append(c)
    return out


def _non_person_tids(tracking_data, results, raw_faces, frame_h,
                     static_px=15.0, short_ratio=0.6):
    """Tracking ids that are NOT people: no face at all AND (rock-static bbox
    OR short/tiny). This works at TRACKER granularity, so a non-person object
    folded into a person's consolidated id by the engine is still caught: its
    boxes match a non-person tracking id and get dropped box-by-box (real
    case: the side chair shares Prajna's tracker/engine id at cam2 frames
    100-125/219-295 because DeepSORT handed the chair's detection to the
    person's id). Returns a set of tracking ids (strings)."""
    # Which tracking ids ever produced a face (raw_faces is keyed by the
    # engine's per-frame id, so match tracking boxes -> reid boxes by IoU).
    face_tids = set()
    tid_stats = {}
    for fid_str, boxes in tracking_data.items():
        fid = int(fid_str)
        faces = raw_faces.get(fid)
        people = results.get(fid_str, [])
        for t in boxes:
            tid = t.get("id")
            bb = t.get("bbox")
            if tid is None or not bb:
                continue
            s = tid_stats.setdefault(tid, {"cx": [], "cy": [], "w": [], "h": []})
            x0, y0, x1, y1 = (float(v) for v in bb[:4])
            s["cx"].append((x0 + x1) / 2.0)
            s["cy"].append((y0 + y1) / 2.0)
            s["w"].append(x1 - x0)
            s["h"].append(y1 - y0)
            if faces and tid not in face_tids:
                # raw_faces is keyed by the engine's per-frame id when it came
                # from the pipeline cache, but by the tracking id on the
                # standalone re-extract fallback — cover both.
                eng = max(people, default=None,
                          key=lambda p: _box_iou(bb, p.get("bbox")))
                if eng is not None and (eng.get("id") in faces or tid in faces):
                    face_tids.add(tid)

    out = set()
    for tid, s in tid_stats.items():
        if tid in face_tids or len(s["cx"]) < 2:
            continue
        static = (max(s["cx"]) - min(s["cx"]) < static_px
                  and max(s["cy"]) - min(s["cy"]) < static_px
                  and max(s["w"]) - min(s["w"]) < static_px
                  and max(s["h"]) - min(s["h"]) < static_px)
        short = max(s["h"]) < short_ratio * frame_h
        # A no-face box that spans nearly the FULL viewport height (>= 95%)
        # is a detector false positive, not a person crop — people crops keep
        # a margin at the frame edges even up close. Real case: a right-edge
        # artefact box [1112,5,1280,708] in a 720-tall frame rendered as an
        # extra fake identity.
        oversized = max(s["h"]) >= 0.95 * frame_h
        if static or short or oversized:
            out.add(tid)
    return out


def _drop_non_person_boxes(results, tracking_data, non_person_tids, min_iou=0.5):
    """Remove reid boxes that overlap a non-person tracking id's box at the
    same frame, whatever consolidated id they were folded into. Returns the
    number of boxes dropped."""
    dropped = 0
    for fid_str in [k for k in results if not k.startswith("_")]:
        tracks = tracking_data.get(fid_str, [])
        if not tracks:
            continue
        people = results.get(fid_str)
        if not people:
            continue
        kept = []
        for p in people:
            bb = p.get("bbox")
            c = p.get("consolidated_id")
            if c is None or c == -1 or not bb:
                kept.append(p)
                continue
            best_tid, best_iou = None, 0.0
            for t in tracks:
                iou = _box_iou(bb, t.get("bbox"))
                if iou > best_iou:
                    best_tid, best_iou = t.get("id"), iou
            if best_tid is not None and best_iou >= min_iou \
                    and best_tid in non_person_tids:
                dropped += 1
                continue
            kept.append(p)
        results[fid_str] = kept
    return dropped


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

    with open(tracking_json_path, encoding="utf-8") as f:
        tracking_data = json.load(f)
    cap = cv2.VideoCapture(video_path)
    cap_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720.0
    cap.release()

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
        raw_faces = _collect_faces(video_path, tracking_data, frame_ids, stride, extractor)

    if not raw_faces:
        logger.warning("face_verify: no faces found — leaving output unchanged")
        results["__tracks__"] = old_tracks
        return None

    # Box-level non-person suppression: drop no-face static/tiny tracker boxes
    # even when the engine folded them into a person's consolidated id (the
    # side chair rendering under Prajna's id). Runs before any face voting so
    # the dropped boxes never distort cid stats.
    non_person_tids = _non_person_tids(tracking_data, results, raw_faces, cap_h)
    if non_person_tids:
        n_dropped = _drop_non_person_boxes(results, tracking_data, non_person_tids)
        logger.info("face_verify: dropped %d non-person box(es) "
                    "(no faces, static/tiny tracking id): %s",
                    n_dropped, sorted(non_person_tids))

    classifier = _FaceClassifier(identity_db)
    classified = {
        fid: {tid: classifier.classify(face) for tid, face in boxes.items()}
        for fid, boxes in raw_faces.items()
    }

    n_swap = _detect_and_fix_swaps(classified, results)
    engine_names = {
        int(c): t.get("name")
        for c, t in old_tracks.items()
        if t.get("name")
    }
    n_split, seeds = _split_person_switches(classified, results, engine_names)
    n_false, seeds2 = _split_false_merges(classified, results, raw_faces)
    seeds.update(seeds2)
    n_merge, merged_names = _merge_fragments(classified, results, seeds, raw_faces)

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
        if cid in seeds and seeds[cid] is None:
            # Split-created UNIDENTIFIED identity: the splitter set it apart
            # because the relative face tally says it is a DIFFERENT person
            # (unregistered 3rd person). Do not let a couple of stray faces
            # that happen to score >= 0.50 against a registered gallery rename
            # it and re-fold it into the person it was split from.
            decision[cid] = (None, None, None, [])
            continue
        d = _face_decided(cid)
        if d is not None:
            decision[cid] = d
        elif cid in seeds:
            # split-created identity: name comes from the splitter's face winner
            decision[cid] = (seeds[cid], None, None, ["face"])
        elif cid in merged_names:
            # fragment root: name from the face-vote agreement that merged the
            # fragments (see _merge_fragments). Prefer a preserved old name that
            # the DB still confirms, else the merged agreement.
            if old.get("name") and (old.get("face_sim") or 0) >= _DB_FACE_CONFIRMED:
                decision[cid] = (old.get("name"), old.get("similarity"),
                                 old.get("face_sim"), old.get("cues", []))
            else:
                decision[cid] = (merged_names[cid], None, None, ["face"])
        elif old.get("name") and not old.get("cues") \
                and old.get("similarity") is None:
            # previously user-corrected in the UI (no engine metrics)
            decision[cid] = (old.get("name"), old.get("similarity"),
                             old.get("face_sim"), old.get("cues", []))
        elif old.get("name") and (old.get("face_sim") or 0) >= _DB_FACE_CONFIRMED:
            # engine face-named with a face the DB itself confirms
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

    # Drop non-people (the static side chair, tiny edge slivers) that have no
    # faces, so they never appear as a fake id in the summary or overlay.
    suppressed = set(_non_person_cids(results, cid_faces, cap_h))
    if suppressed:
        for fid_str in [k for k in results if not k.startswith("_")]:
            kept = [p for p in results[fid_str]
                    if p.get("consolidated_id") not in suppressed]
            results[fid_str] = kept
        logger.info("face_verify: dropped %s non-person track(s) "
                    "(no faces, static/tiny bbox): %s",
                    len(suppressed), sorted(suppressed))

    tracks_payload = {}
    for cid in remap.values():
        if cid in suppressed:
            continue
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
        json.dump(results, f, separators=(",", ":"))

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
    logger.info("face_verify: %s swap(s), %s split(s), %s false-merge split(s), "
                "%s merge(s); %s/%s identities named", n_swap, n_split, n_false,
                n_merge, named, len(tracks_payload))
    return people
