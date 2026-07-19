"""
Validation script for the Multi-Camera module.

Run this FIRST after unzipping multicamera/ into your repo, before trying
a full python run_multicam.py. It checks everything that commonly goes
wrong, in order, and tells you exactly what's missing.

Usage:
    python multicamera/validate_setup.py
"""

import sys
import importlib
from pathlib import Path

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

errors = []
warnings = []


def check(label, condition, fix_hint=""):
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}")
        if fix_hint:
            print(f"   -> {fix_hint}")
        errors.append(label)


def warn(label, condition, hint=""):
    if not condition:
        print(f"{WARN}{label}")
        if hint:
            print(f"   -> {hint}")
        warnings.append(label)


def main():
    print("=" * 60)
    print("Multi-Camera Module — Environment Validation")
    print("=" * 60)

    # 1. Are we running from the repo root?
    repo_markers = ["main.py", "config.py", "detection", "tracking"]
    at_root = all(Path(m).exists() for m in repo_markers)
    check(
        "Running from repo root",
        at_root,
        "cd into the folder containing main.py / config.py before running this."
    )
    if not at_root:
        print("\nStopping early — fix the above and re-run.")
        sys.exit(1)

    # 2. Module files present
    for f in [
        "multicamera/__init__.py",
        "multicamera/camera_config.py",
        "multicamera/stream_manager.py",
        "multicamera/multi_cam_pipeline.py",
        "run_multicam.py",
    ]:
        check(f"{f} present", Path(f).exists())

    # 3. Required packages
    for pkg in ["cv2", "torch", "ultralytics", "deep_sort_realtime"]:
        try:
            importlib.import_module(pkg)
            check(f"Package '{pkg}' importable", True)
        except ImportError:
            check(
                f"Package '{pkg}' importable", False,
                "pip install -r requirements.txt --break-system-packages"
            )

    # 4. GPU availability (informational only — CPU is fine, just slower)
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        warn(
            "CUDA GPU not available (will run on CPU — much slower)",
            cuda_ok,
        )
    except ImportError:
        pass

    # 5. YOLO weights present
    check(
        "models/yolov8s.pt present",
        Path("models/yolov8s.pt").exists(),
        "Download YOLOv8s weights and place at models/yolov8s.pt "
        "(see README.md 'Required Assets' section)."
    )

    # 6. Camera sources exist
    try:
        sys.path.insert(0, ".")
        from multicamera.camera_config import CAMERAS
        if not CAMERAS:
            check("At least one camera configured", False)
        for cam in CAMERAS:
            src = cam["source"]
            if isinstance(src, str):
                check(
                    f"Camera '{cam['camera_id']}' source exists: {src}",
                    Path(src).exists(),
                    f"Add a video file at {src}, or edit "
                    f"multicamera/camera_config.py to point at a real file."
                )
            else:
                warn(f"Camera '{cam['camera_id']}' uses a non-file source "
                     f"({src}) — can't pre-check webcam/RTSP sources here.")
    except Exception as e:
        check("multicamera/camera_config.py imports cleanly", False, str(e))

    # 7. Duplicate camera_id check
    try:
        ids = [c["camera_id"] for c in CAMERAS]
        check("No duplicate camera_id values", len(ids) == len(set(ids)))
    except NameError:
        pass

    print("=" * 60)
    if errors:
        print(f"{FAIL} {len(errors)} problem(s) found — fix these before running run_multicam.py")
        sys.exit(1)
    else:
        print(f"{PASS} All checks passed. You're ready for a real run, e.g.:")
        print("   python run_multicam.py --max-frames 50 --device cpu")
    if warnings:
        print(f"{WARN}{len(warnings)} warning(s) — not blocking, but worth checking")
    print("=" * 60)


if __name__ == "__main__":
    main()
