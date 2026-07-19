"""
Standalone entry point for the Registration & Identity Database module.

Kept SEPARATE from main.py / run_multicam.py on purpose — a new top-level
script means zero merge-conflict risk with teammates. Once every module is
ready, integration can import `IdentityDatabase.export_for_reid()` from
here into whatever ties Phase 3 together.

Usage:
    # Register one person from a folder of their photos
    python register.py add --name "Alice" --images photos/alice/*.jpg

    # Register everyone at once from a folder-of-folders layout:
    #   known_persons/Alice/*.jpg
    #   known_persons/Bob/*.jpg
    python register.py bulk --dir known_persons

    # List everyone currently registered
    python register.py list

    # Search by (partial) name
    python register.py search --name ali

    # Sanity-check environment first
    python registration/validate_setup.py
"""

import argparse
import glob
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_add(args):
    from registration.register_person import register_person

    image_paths = []
    for pattern in args.images:
        matches = glob.glob(pattern)
        image_paths.extend(matches if matches else [pattern])

    record = register_person(args.name, image_paths)
    print(f"\n✅ Registered '{args.name}'")
    print(f"   Images used : {record['metadata']['num_images']}")
    print(f"   Registered  : {record['metadata']['registered_at']}")


def cmd_bulk(args):
    from registration.register_person import register_person
    from registration.identity_db import IdentityDatabase

    root = Path(args.dir)
    if not root.exists():
        print(f"❌ Directory not found: {root}")
        return 1

    db = IdentityDatabase()
    person_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not person_dirs:
        print(f"❌ No person sub-folders found under {root}")
        print("   Expected layout: known_persons/<name>/*.jpg")
        return 1

    for person_dir in person_dirs:
        images = [
            str(p) for p in person_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if not images:
            logger.warning(f"Skipping '{person_dir.name}': no images found")
            continue
        try:
            register_person(person_dir.name, images, db=db)
        except ValueError as e:
            logger.warning(f"Skipping '{person_dir.name}': {e}")

    print(f"\n✅ Bulk registration complete. {len(db)} person(s) in the database.")


def cmd_list(args):
    from registration.identity_db import IdentityDatabase

    db = IdentityDatabase()
    names = db.list_persons()
    if not names:
        print("Database is empty. Register someone with 'register.py add' first.")
        return
    print(f"{len(names)} registered person(s):")
    for name in names:
        record = db.get_person(name)
        print(f"  - {name}  ({record['metadata']['num_images']} image(s), "
              f"registered {record['metadata']['registered_at']})")


def cmd_search(args):
    from registration.identity_db import IdentityDatabase

    db = IdentityDatabase()
    results = db.search_by_name(args.name)
    if not results:
        print(f"No match for '{args.name}'")
        return
    for name, record in results:
        print(f"  - {name}  ({record['metadata']['num_images']} image(s))")


def main():
    parser = argparse.ArgumentParser(description="Person Registration & Identity Database")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Register one person from image files")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--images", nargs="+", required=True,
                        help="Image file paths or glob patterns")
    p_add.set_defaults(func=cmd_add)

    p_bulk = sub.add_parser("bulk", help="Register everyone under a folder-of-folders")
    p_bulk.add_argument("--dir", required=True,
                         help="Folder containing one sub-folder of images per person")
    p_bulk.set_defaults(func=cmd_bulk)

    p_list = sub.add_parser("list", help="List all registered persons")
    p_list.set_defaults(func=cmd_list)

    p_search = sub.add_parser("search", help="Search registered persons by (partial) name")
    p_search.add_argument("--name", required=True)
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    main()
