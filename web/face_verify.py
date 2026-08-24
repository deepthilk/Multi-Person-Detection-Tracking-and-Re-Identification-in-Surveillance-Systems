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

    cid_tids = {}
    for (c, t), (a, b) in tid_info.items():
        cid_tids.setdefault(c, []).append(t)

    splits = 0
    seeds = {}
    for cid, tids in sorted(cid_tids.items(), key=lambda kv: kv[0]):
        tids = sorted(tids, key=int)
        if len(tids) < 2:
            continue
        groups, no_winner = {}, []
        for t in tids:
            w = _winner(tid_votes.get((cid, t), {}))
            (groups.setdefault(w, []).append(t) if w is not None
             else no_winner.append(t))
        if len(groups) >= 2:
            # Classic false merge: multiple trackers each decided a DIFFERENT
            # person. Split every later group off (earliest keeps the cid).
            ordered = sorted(groups.items(),
                             key=lambda kv: min(tid_info[(cid, t)][0] for t in kv[1]))
            for winner, gtids in ordered[1:]:
                new_cid = _mint_split(cid, gtids)
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
    return splits, seeds


def _merge_fragments(classified, results, seeds=None):
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

    The gate is the FACE-VOTE NAME, preferring the DECISIVE name computed
    exactly as the final naming does (see _face_decided), then the SPLITTER'S
    SEED name (a split-created identity's name is the splitter's face winner),
    then the RELATIVE best-match winner — the same weak-face evidence the
    false-merge splitter uses (each face votes for the gallery it looks most
    like, no absolute score gate). Two identities are merged when they agree on
    a name AND at least one side is STRONG (decisive name or splitter seed):

      • genuine fragments merge — e.g. two "usha" fragments where Usha's faces
        only ever reach the relative bar, or a relative "deeps" fragment folded
        into a decisive "deeps" fragment;
      • different people NEVER merge on weak evidence alone — a relative-only
        vote (both sides weak) is not enough, so two people who merely look
        like the same low-res gallery blob stay separate;
      • the engine's own name is NEVER trusted: identical uniforms make body
        appearance ~equal, so the engine routinely merges DIFFERENT people into
        one identity (the splitter then un-merges them), and a split-created
        identity has no engine name at all.

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
        """Best-match-name majority with no absolute score gate (same weak-face
        evidence as the false-merge splitter's relative tally)."""
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

    merged_pairs = 0
    for i, a in enumerate(cids):
        for b in cids[i + 1:]:
            if find(a) == find(b):
                continue
            na, sa = _name(a)
            nb, sb = _name(b)
            if not (na and nb and na == nb):
                continue            # only the same face-vote name confirms identity
            if not (sa or sb):
                continue            # never merge two weak-only fragments
            union(a, b)
            merged_pairs += 1
            logger.info("face_verify: MERGE fragmented ids %s + %s "
                        "(both '%s')", a, b, na)

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
    engine_names = {
        int(c): t.get("name")
        for c, t in old_tracks.items()
        if t.get("name")
    }
    n_split, seeds = _split_person_switches(classified, results, engine_names)
    n_false, seeds2 = _split_false_merges(classified, results, raw_faces)
    seeds.update(seeds2)
    n_merge, merged_names = _merge_fragments(classified, results, seeds)

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
