"""
Tests for the web-facing registration contract hardening:

  * upload validation (_save_upload: extension, empty, non-image)
  * form-field clamping (_clamp: augmentations / samples / top bounds)
  * auto-fix robustness against unreadable/empty videos
    (scan_video_for_candidates must return [] instead of crashing)

The API validation functions are tested directly (no live server needed);
only OpenCV + fastapi's HTTPException are required.

Run with:
    python -m registration.tests.test_web_contract
"""

import io
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import HTTPException

from web import registration_api as api
from registration.autofix import scan_video_for_candidates, scan_videos_for_all_persons


class _FakeUploadFile:
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self.file = io.BytesIO(data)


def _png_bytes(size: int = 32) -> bytes:
    import cv2
    img = np.random.default_rng(1).integers(0, 256, (size, size, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".png", img)
    assert ok
    return enc.tobytes()


# ── _clamp ──────────────────────────────────────────────────────────────────

def test_clamp_bounds():
    assert api._clamp(5, 0, 30, "augmentations") == 5
    assert api._clamp(0, 0, 30, "augmentations") == 0
    try:
        api._clamp(31, 0, 30, "augmentations")
        assert False, "above upper bound must raise"
    except HTTPException as e:
        assert e.status_code == 400
    try:
        api._clamp(-1, 1, 200, "samples")
        assert False, "below lower bound must raise"
    except HTTPException as e:
        assert e.status_code == 400
    try:
        api._clamp("many", 0, 30, "augmentations")
        assert False, "non-integer must raise"
    except HTTPException as e:
        assert e.status_code == 400
    print(" test_clamp_bounds passed")


# ── upload validation ───────────────────────────────────────────────────────

def test_save_upload_rejects_bad_inputs():
    tmp = Path(tempfile.mkdtemp())
    old_upload_dir = api.UPLOAD_DIR
    api.UPLOAD_DIR = tmp
    try:
        # wrong extension
        try:
            api._save_upload(_FakeUploadFile("photo.txt", b"x"), "reg")
            assert False, "wrong extension must raise"
        except HTTPException as e:
            assert e.status_code == 400

        # right extension but empty file
        try:
            api._save_upload(_FakeUploadFile("empty.png", b""), "reg")
            assert False, "empty file must raise"
        except HTTPException as e:
            assert e.status_code == 400

        # right extension but not actually an image
        try:
            api._save_upload(_FakeUploadFile("fake.jpg", b"this is text, not jpeg data"), "reg")
            assert False, "unreadable image must raise"
        except HTTPException as e:
            assert e.status_code == 422

        assert list(tmp.iterdir()) == [], "rejected uploads must not leave files behind"

        # valid PNG passes and is saved
        saved = api._save_upload(_FakeUploadFile("real.png", _png_bytes()), "reg")
        assert saved.exists() and saved.stat().st_size > 0
        assert list(tmp.iterdir()), "accepted upload should be on disk"
        print(" test_save_upload_rejects_bad_inputs passed")
    finally:
        api.UPLOAD_DIR = old_upload_dir
        shutil.rmtree(tmp)


# ── auto-fix robustness against bad videos ──────────────────────────────────

def test_autofix_skips_bad_videos():
    tmp = Path(tempfile.mkdtemp())
    try:
        missing = str(tmp / "does_not_exist.mp4")
        ref = np.random.default_rng(0).normal(size=698).astype(np.float32)
        ref /= np.linalg.norm(ref)

        # single-video scan: unreadable video -> empty candidate list, no crash
        out = scan_video_for_candidates(missing, detector=None, ref_avg=ref,
                                        out_dir=tmp, samples=5)
        assert out == [], "unreadable video must yield no candidates (not crash)"

        # garbage file that opencv cannot decode
        garbage = tmp / "garbage.mp4"
        garbage.write_bytes(b"\x00\x00\x00\x00not a video")
        out = scan_video_for_candidates(str(garbage), detector=None, ref_avg=ref,
                                        out_dir=tmp, samples=5)
        assert out == []

        # multi-person scan with a bad video in the list: best[name] stays empty
        refs = {"Alice": ref}
        best = scan_videos_for_all_persons([missing], detector=None, ref_embeddings=refs,
                                           out_dir=tmp, samples=5)
        assert best["Alice"] == []
        print(" test_autofix_skips_bad_videos passed")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_clamp_bounds()
    test_save_upload_rejects_bad_inputs()
    test_autofix_skips_bad_videos()
    print("\nAll web-contract tests passed ")
