"""
Validation script for the Registration module.

Run this FIRST after adding registration/ to your repo, before trying
register.py for real.

Usage:
    python registration/validate_setup.py
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
    print("Registration Module — Environment Validation")
    print("=" * 60)

    # 1. Running from repo root?
    repo_markers = ["main.py", "config.py", "reidentification"]
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
        "registration/__init__.py",
        "registration/db_config.py",
        "registration/embedder.py",
        "registration/identity_db.py",
        "registration/register_person.py",
        "register.py",
    ]:
        check(f"{f} present", Path(f).exists())

    # 3. Required packages (all already in requirements.txt — no new deps)
    for pkg in ["cv2", "torch", "torchvision", "numpy", "PIL"]:
        try:
            importlib.import_module(pkg)
            check(f"Package '{pkg}' importable", True)
        except ImportError:
            check(
                f"Package '{pkg}' importable", False,
                "pip install -r requirements.txt --break-system-packages"
            )

    # 4. Can we see Deepthi's Re-ID engine? (imported, never modified)
    try:
        from reidentification.reid_main import ReIDEngine  # noqa: F401
        check("reidentification.reid_main.ReIDEngine importable", True)
    except Exception as e:
        check(
            "reidentification.reid_main.ReIDEngine importable", False,
            f"Make sure reidentification/reid_main.py exists and its deps "
            f"are installed. Error: {e}"
        )

    # 5. Output dirs
    from registration.db_config import DB_SETTINGS
    Path(DB_SETTINGS["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(DB_SETTINGS["images_dir"]).mkdir(parents=True, exist_ok=True)
    check("Output directories created", True)

    print("=" * 60)
    if errors:
        print(f"{FAIL} {len(errors)} problem(s) found — fix these before running register.py")
        sys.exit(1)
    else:
        print(f"{PASS} All checks passed. You're ready, e.g.:")
        print('   python register.py add --name "Alice" --images photo1.jpg photo2.jpg')
    if warnings:
        print(f"{WARN}{len(warnings)} warning(s) — not blocking, but worth checking")
    print("=" * 60)


if __name__ == "__main__":
    main()
