"""
Regression tests for face-primary name resolution (web/multicam_pipeline).

The rule under test: if a face was EVER detected on an identity's tracks, the
face decides the name; an inconclusive face must never be overruled by a body
guess (that body-over-face fallback mislabelled a stranger as "Pranjali" with a
~0.31 face). The body is only consulted when the identity had no face at all.

These tests import only the pure helper — torch/onnx models load lazily inside
the pipeline, so nothing heavy is pulled in here.
"""

import numpy as np

from web.multicam_pipeline import _resolve_identity_name


def _sim(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _db(records):
    class _DB:
        def __init__(self, data):
            self._data = data
            self.body_calls = []

        def match(self, query_embedding, query_face_embedding=None, top_k=None, threshold=None):
            # A deliberately strong body match: if the face is present but
            # inconclusive, the helper must NOT consult this at all.
            self.body_calls.append(threshold)
            return [("BODY-MATCH", 0.99)]

    return _DB(records)


def _face(frame_id, vec):
    return (frame_id, [0, 0, 10, 10], np.asarray(vec, dtype=np.float64), 0.9, [0, 0, 5, 5])


def test_face_confirm_names_person_with_best_frame():
    records = {
        "Alice": {"average_face_descriptor": [1.0, 0.0]},
        "Bob": {"average_face_descriptor": [0.0, 1.0]},
    }
    db = _db(records)
    # Two candidate frames: frame 5 clearly Alice, frame 8 weakly Alice.
    face_candidates = {
        10: [_face(5, [0.99, 0.01]), _face(8, [0.62, 0.38])],
    }
    result = _resolve_identity_name(
        1, None, face_candidates, {10: 1}, db, face_confirm=0.40,
        match_threshold=0.55, face_similarity=_sim,
    )
    assert result is not None
    assert result[0] == "Alice"
    assert result[1] > 0.9
    assert db.body_calls == []  # face decided; body never consulted


def test_inconclusive_face_stays_unknown_not_body():
    """Regression for Failure 2: a ~0.31 face must NOT be overruled by a strong
    body match (which previously named a stranger 'Pranjali')."""
    records = {
        "Alice": {"average_face_descriptor": [1.0, 0.0]},
    }
    db = _db(records)
    face_candidates = {10: [_face(3, [0.30, 0.95])]}  # cosine ~0.30, below the 0.40 confirm bar
    result = _resolve_identity_name(
        1, None, face_candidates, {10: 1}, db, face_confirm=0.40,
        match_threshold=0.55, face_similarity=_sim,
    )
    assert result is None
    assert db.body_calls == []  # the body guess must never fire over a real face


def test_no_face_at_all_falls_back_to_body():
    records = {"Alice": {"average_face_descriptor": [1.0, 0.0]}}
    db = _db(records)
    result = _resolve_identity_name(
        1, np.zeros(2), {}, {}, db, face_confirm=0.40,
        match_threshold=0.55, face_similarity=_sim,
    )
    assert result == ("BODY-MATCH", 0.99)
    assert db.body_calls == [0.55]


def test_no_face_and_weak_body_stays_unknown():
    class _DB:
        _data = {}

        def match(self, query_embedding, query_face_embedding=None, top_k=None, threshold=None):
            return []

    result = _resolve_identity_name(
        1, np.zeros(2), {}, {}, _DB(), face_confirm=0.40,
        match_threshold=0.55, face_similarity=_sim,
    )
    assert result is None


def test_face_less_gallery_person_cannot_be_confirmed():
    """A registered person with no average face descriptor (e.g. Deepthi's
    body-only photos) can never be named by a face, even on a face-on track."""
    records = {
        "Deepthi": {"average_face_descriptor": None},
    }
    db = _db(records)
    face_candidates = {10: [_face(1, [0.99, 0.0])]}
    result = _resolve_identity_name(
        1, None, face_candidates, {10: 1}, db, face_confirm=0.40,
        match_threshold=0.55, face_similarity=_sim,
    )
    assert result is None
    assert db.body_calls == []


def test_candidates_from_other_identities_are_ignored():
    records = {"Alice": {"average_face_descriptor": [1.0, 0.0]}}
    db = _db(records)
    face_candidates = {
        10: [_face(1, [0.99, 0.01])],   # belongs to sid 1
        99: [_face(2, [0.99, 0.01])],   # belongs to sid 2, not sid 1
    }
    result = _resolve_identity_name(
        1, None, face_candidates, {10: 1, 99: 2}, db, face_confirm=0.40,
        match_threshold=0.55, face_similarity=_sim,
    )
    assert result is not None and result[0] == "Alice"
