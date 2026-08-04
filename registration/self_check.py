"""
Automated self-audit for the identity database.

Answers three questions a reviewer will ask of any registration dataset:

  1. What is in the database per person? (photo count, face descriptors
     present or not)
  2. Can the cues tell the registrants apart? (body and face cross-similarity
     matrices, with pairs above the guardrail thresholds flagged)
  3. Is the database self-consistent? (every registered photo, re-embedded
     deterministically, must match back to its own person as top-1)

IMPORTANT — what this is, and is not: this is a *closed-loop* consistency
check. The photos re-embedded here are the same photos that built the
database, and registration is deterministic (fixed RNG seed), so a high
top-1 rate is EXPECTED. It proves the database is internally coherent and
reproducible; it does NOT prove recognition on unseen surveillance footage.
Cross-domain evidence comes from the live Re-ID pipeline (matching against
actual video tracks), not from this report. The report says so explicitly so
it cannot be mistaken for an evaluation result.

Usage:
    python register.py self-check [--out reports/self_check.json]
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from registration.db_config import DB_SETTINGS, GUARDRAIL_SETTINGS
from registration.identity_db import IdentityDatabase

logger = logging.getLogger(__name__)


def _per_person_stats(db: IdentityDatabase) -> list:
    """Per-person inventory: photo count, how many yielded a face, and
    whether an average face descriptor exists for matching."""
    stats = []
    for name in db.list_persons():
        record = db.get_person(name)
        num_faces = len(record.get("face_embeddings", []))
        stats.append({
            "name": name,
            "num_images": record["metadata"].get("num_images", len(record.get("embeddings", []))),
            "num_faces": num_faces,
            "has_face_descriptor": record.get("average_face_descriptor") is not None,
        })
    return stats


def _cross_sim_matrix(db: IdentityDatabase, face: bool = False) -> dict:
    """NxN average-descriptor similarity matrix. Diagonal omitted (it is
    trivially 1.0). face=False uses the body cue, face=True the face cue
    (missing pairs are omitted)."""
    matrix = {}
    for name in db.list_persons():
        row = {}
        for other in db.list_persons():
            if other == name:
                continue
            sim = _pair_sim(db, name, other, face=face)
            if sim is not None:
                row[other] = round(sim, 4)
        matrix[name] = row
    return matrix


def _pair_sim(db: IdentityDatabase, name_a: str, name_b: str, face: bool):
    rec_a, rec_b = db.get_person(name_a), db.get_person(name_b)
    if face:
        fa, fb = rec_a.get("average_face_descriptor"), rec_b.get("average_face_descriptor")
        if fa is None or fb is None:
            return None
        from reidentification.face_cue import FaceCueExtractor
        return FaceCueExtractor.similarity(fa, fb)
    return _cosine(rec_a["average_embedding"], rec_b["average_embedding"])


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def _extract_face(path: str):
    """Best-effort face descriptor for one stored photo. Returns None when
    no face is found (body-only match path, like live inference)."""
    try:
        import cv2
        from reidentification.face_cue import FaceCueExtractor
        img = cv2.imread(path)
        if img is None:
            return None
        return FaceCueExtractor().extract(img, [0, 0, img.shape[1], img.shape[0]])
    except Exception as e:
        logger.warning(f"Face extraction failed for {path}: {e}")
        return None


def _self_consistency(db: IdentityDatabase) -> dict:
    """Re-embed every registered photo (deterministically) and confirm it
    matches back to its own person as top-1 via IdentityDatabase.match() —
    the same matcher used by live Re-ID."""
    from registration.embedder import embed_image

    per_photo = []
    total = correct = 0
    per_person = {}

    for name in db.list_persons():
        record = db.get_person(name)
        ok = 0
        person_total = 0
        for i, emb in enumerate(record.get("embeddings", [])):
            path = record["metadata"].get("image_paths", [])
            path = path[i] if i < len(path) else None
            if path is None or not Path(path).exists():
                continue

            query = embed_image(path)          # deterministic (fixed seed)
            face_feat = _extract_face(path)    # None => body-only path
            if query is None:
                continue

            top1 = db.match(query, query_face_embedding=face_feat, top_k=1)
            top1_name = top1[0][0] if top1 else None
            top1_sim = top1[0][1] if top1 else 0.0
            is_correct = top1_name == name

            total += 1
            correct += int(is_correct)
            ok += int(is_correct)
            person_total += 1

            per_photo.append({
                "person": name,
                "photo": str(path),
                "top1": top1_name,
                "top1_sim": round(float(top1_sim), 4),
                "correct": is_correct,
                "face_used": face_feat is not None,
            })

        per_person[name] = {"correct": ok, "total": person_total}

    return {
        "per_photo": per_photo,
        "summary": {
            "total_photos": total,
            "correct_top1": correct,
            "top1_accuracy": round(correct / total, 4) if total else None,
            "per_person": {n: {"correct": v["correct"], "total": v["total"]}
                           for n, v in per_person.items()},
        },
    }


def _guardrail_flags(body_matrix: dict, face_matrix: dict) -> dict:
    body_bar = GUARDRAIL_SETTINGS["body_discrimination_threshold"]
    face_bar = GUARDRAIL_SETTINGS["face_discrimination_threshold"]

    body_pairs = [
        {"a": a, "b": b, "body_sim": sim}
        for a, row in body_matrix.items()
        for b, sim in row.items() if a < b and sim >= body_bar
    ]
    face_pairs = []
    for a, row in face_matrix.items():
        for b, sim in row.items():
            if a < b and sim >= face_bar:
                face_pairs.append({"a": a, "b": b, "face_sim": sim})
    return {
        "body_bar": body_bar,
        "face_bar": face_bar,
        "body_pairs_above_bar": sorted(body_pairs, key=lambda x: x["body_sim"], reverse=True),
        "face_pairs_above_bar": sorted(face_pairs, key=lambda x: x["face_sim"], reverse=True),
    }


def run_self_check(db: IdentityDatabase = None, out_path: str = None) -> dict:
    """Produce the full self-audit report (dict). If out_path is given, also
    writes the JSON report there."""
    # Same None-guard as register_person: an empty IdentityDatabase is falsy
    # (it defines __len__), so `db or IdentityDatabase()` would discard a
    # caller-supplied empty db.
    db = db if db is not None else IdentityDatabase()

    report = {
        "generated_at": datetime.now().isoformat(),
        "db_path": str(db.db_path),
        "n_persons": len(db),
        "persons": db.list_persons(),
        "per_person": _per_person_stats(db),
        "body_cross_sim_matrix": _cross_sim_matrix(db, face=False),
        "face_cross_sim_matrix": _cross_sim_matrix(db, face=True),
        "self_consistency": _self_consistency(db),
        "guardrail_flags": _guardrail_flags(
            _cross_sim_matrix(db, face=False),
            _cross_sim_matrix(db, face=True),
        ),
        "caveat": (
            "Closed-loop self-consistency only: the re-embedded photos are the "
            "same ones that built the database, and registration is deterministic, "
            "so high top-1 accuracy here is expected. It does NOT measure "
            "recognition on unseen footage; cross-domain evidence comes from the "
            "live Re-ID pipeline against actual video tracks."
        ),
    }

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Self-check report written to {out}")

    return report


def print_report(report: dict):
    """Human-readable rendering of the report for the console."""
    print("\n" + "=" * 72)
    print("IDENTITY DATABASE SELF-CHECK")
    print("=" * 72)
    print(f"Database : {report['db_path']}  ({report['n_persons']} person(s))")
    print(f"Generated: {report['generated_at']}")

    print("\n-- per-person inventory --------------------------------")
    for p in report["per_person"]:
        face = "yes" if p["has_face_descriptor"] else "NO (body-only)"
        print(f"  {p['name']:<12s} {p['num_images']} photo(s), {p['num_faces']} face(s) "
              f"[face descriptor: {face}]")

    print("\n-- body cross-similarity (average embeddings) -----------")
    _print_matrix(report["body_cross_sim_matrix"])
    print("\n-- face cross-similarity (average face descriptors) -----")
    _print_matrix(report["face_cross_sim_matrix"])

    print("\n-- guardrail flags --------------------------------------")
    gf = report["guardrail_flags"]
    if not gf["body_pairs_above_bar"] and not gf["face_pairs_above_bar"]:
        print("  None — every registrant pair is separated on both cues.")
    for pair in gf["body_pairs_above_bar"]:
        print(f"  BODY  {pair['a']} <-> {pair['b']}: sim {pair['body_sim']:.3f} "
              f">= {gf['body_bar']} (appearance cannot separate them)")
    for pair in gf["face_pairs_above_bar"]:
        print(f"  FACE  {pair['a']} <-> {pair['b']}: sim {pair['face_sim']:.3f} "
              f">= {gf['face_bar']} (possible duplicate registration)")

    print("\n-- closed-loop self-consistency -------------------------")
    s = report["self_consistency"]["summary"]
    if s["top1_accuracy"] is None:
        print("  No stored photos could be re-embedded (empty DB or missing files).")
    else:
        print(f"  {s['correct_top1']}/{s['total_photos']} photos matched back to their "
              f"own person as top-1 (accuracy {s['top1_accuracy']:.1%})")
        for name, cnt in sorted(s["per_person"].items()):
            print(f"    {name:<12s} {cnt['correct']}/{cnt['total']}")

        failures = [p for p in report["self_consistency"]["per_photo"] if not p["correct"]]
        if failures:
            print("  Mismatches (need investigation):")
            for p in failures:
                print(f"    {p['person']} photo {p['photo']} -> matched "
                      f"{p['top1']} (sim {p['top1_sim']:.3f}, face={'yes' if p['face_used'] else 'no'})")

    print("\n" + "=" * 72)
    print("NOTE: this is a closed-loop consistency check, not a benchmark on")
    print("unseen footage. It shows the DB is internally coherent; cross-domain")
    print("evidence comes from the live Re-ID pipeline on real video tracks.")
    print("=" * 72 + "\n")


def _print_matrix(matrix: dict):
    if not matrix:
        print("  (empty)")
        return
    names = sorted(matrix.keys())
    print("  " + "".join(f"{n:>10s}" for n in names))
    for a in names:
        row = matrix.get(a, {})
        cells = []
        for b in names:
            if b == a:
                cells.append("   --      ")
            else:
                v = row.get(b)
                cells.append(f"{v:>10.3f}" if v is not None else "     .    ")
        print(f"{a:>3s} " + "".join(cells))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="register.py self-check",
        description="Audit the identity database (inventory, cross-similarity, "
                    "closed-loop consistency) and print + save a report.",
    )
    parser.add_argument("--out", default=None,
                        help="Where to write the JSON report (e.g. reports/self_check.json)")
    args = parser.parse_args(argv)
    report = run_self_check(out_path=args.out)
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
