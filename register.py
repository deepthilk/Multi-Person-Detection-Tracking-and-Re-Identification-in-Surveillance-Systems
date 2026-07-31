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

    # Replace a person's photos instead of adding to them
    python register.py add --name "Alice" --images new_photos/*.jpg --overwrite

    # Remove someone
    python register.py delete --name "Bob"

    # Back up / restore the whole database
    python register.py backup --path backups/db_2026-07-27.json
    python register.py restore --path backups/db_2026-07-27.json

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

    record = register_person(args.name, image_paths, overwrite=args.overwrite,
                              num_augmentations=args.augmentations)
    print(f"\n✅ Registered '{args.name}'")
    print(f"   Images used : {record['metadata']['num_images']}")
    print(f"   Augmentations per image : {args.augmentations}")
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
            register_person(person_dir.name, images, db=db, num_augmentations=args.augmentations)
        except ValueError as e:
            logger.warning(f"Skipping '{person_dir.name}': {e}")

    print(f"\n✅ Bulk registration complete. {len(db)} person(s) in the database.")


def cmd_delete(args):
    from registration.identity_db import IdentityDatabase

    db = IdentityDatabase()
    if not db.person_exists(args.name):
        print(f"'{args.name}' is not in the database — nothing to delete.")
        return
    if not args.yes:
        confirm = input(f"Delete '{args.name}' and all their registered photos? [y/N] ")
        if confirm.strip().lower() != "y":
            print("Cancelled.")
            return
    db.delete_person(args.name)
    print(f"✅ Deleted '{args.name}' from the identity database.")


def cmd_backup(args):
    from registration.identity_db import IdentityDatabase

    db = IdentityDatabase()
    db.export_db(args.path)
    print(f"✅ Backed up {len(db)} person(s) -> {args.path}")


def cmd_restore(args):
    from registration.identity_db import IdentityDatabase

    db = IdentityDatabase()
    db.import_db(args.path, merge=not args.replace)
    print(f"✅ Restored from {args.path} ({'replaced' if args.replace else 'merged'}). "
          f"Database now has {len(db)} person(s).")


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


def cmd_from_frame(args):
    """Register a person directly from a video frame crop.

    This is the recommended approach when a high-res photo doesn't match
    the video domain — register the person using their actual appearance
    in the video itself. Since the Re-ID model matches well within the
    video domain (same quality, lighting, resolution), this guarantees
    the person will be recognised during live inference.

    Usage:
      # Find the person first (use --detect to auto-detect)
      python register.py from-frame --name "Person2" --video input/video1.mp4 \\
          --bbox 90 412 264 863

      # Auto-detect all persons in a frame, then register one
      python register.py from-frame --name "Person2" --video input/video1.mp4 \\
          --frame 33 --detect --pick 0
    """
    import cv2
    from registration.register_person import register_person

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = args.frame if args.frame is not None else total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Could not read frame {frame_id} from {args.video}")
        return

    h, w = frame.shape[:2]
    print(f"   Video: {args.video}  ({w}x{h}, {total_frames} frames)")
    print(f"   Frame: {frame_id}")

    if args.detect:
        from detection.detect_module import PersonDetector
        detector = PersonDetector(conf_threshold=0.25)
        detections = detector.detect(frame)
        if not detections:
            print("❌ No persons detected in this frame.")
            return
        print(f"   Detected {len(detections)} person(s):")
        for i, d in enumerate(detections):
            x1, y1, dw, dh, score = d
            print(f"     [{i}] ({x1},{y1},{x1+dw},{y1+dh}) conf={score:.2f} size={dw}x{dh}")

        if args.pick is not None:
            if args.pick < 0 or args.pick >= len(detections):
                print(f"❌ --pick {args.pick} out of range (0-{len(detections)-1})")
                return
            x1, y1, dw, dh, _ = detections[args.pick]
            bbox = [x1, y1, x1 + dw, y1 + dh]
        else:
            return  # just list detections, user can re-run with --pick
    else:
        if args.bbox is None or len(args.bbox) != 4:
            print("❌ --bbox requires 4 values: x1 y1 x2 y2")
            print("   Or use --detect to auto-detect persons in the frame.")
            return
        bbox = args.bbox

    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        print("❌ Invalid bbox — crop is empty")
        return

    crop_path = f"outputs/registration/_frame_crops/{args.name.replace(' ', '_')}_{frame_id}.jpg"
    import os
    os.makedirs(os.path.dirname(crop_path), exist_ok=True)
    cv2.imwrite(crop_path, crop)
    print(f"   Crop saved: {crop_path} ({crop.shape[1]}x{crop.shape[0]})")

    register_person(args.name, [crop_path], overwrite=args.overwrite,
                    num_augmentations=args.augmentations)
    print(f"   ℹ️  Registered from frame {frame_id}. The crop is saved at {crop_path} "
          f"in case you want to use it again later.")


def cmd_verify(args):
    """Verify a registered person's embedding against a test image and report
    the cosine similarity. This lets you check whether the Re-ID pipeline
    will be able to recognise a person in a given video frame *before*
    running the full pipeline.

    A score above SEARCH_SETTINGS['match_threshold'] (currently 0.55) means
    the registration should work for that test image. A lower score suggests
    you may need to:
      - Use a photo that more closely matches the video quality (e.g. a
        screenshot from the video)
      - Upload additional photos of the person in different conditions
      - Increase --augmentations for more robust embedding averaging
    """
    from registration.embedder import embed_image
    from registration.identity_db import IdentityDatabase

    db = IdentityDatabase()
    if not db.person_exists(args.name):
        print(f"❌ '{args.name}' is not in the database.")
        print("   Register them first with: register.py add --name 'X' --images ...")
        return

    record = db.get_person(args.name)
    registered_avg = record["average_embedding"]

    # Embed the test image using the same robust pipeline
    feat = embed_image(args.test_image, num_augmentations=args.augmentations)
    if feat is None:
        print(f"❌ Could not generate embedding for test image: {args.test_image}")
        return

    def _cos(a, b):
        import numpy as np
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))

    from registration.db_config import SEARCH_SETTINGS

    sim = _cos(feat, registered_avg)
    threshold = SEARCH_SETTINGS["match_threshold"]

    print(f"\n📊 Registration verification for '{args.name}'")
    print(f"   Test image      : {args.test_image}")
    print(f"   Cosine similarity: {sim:.4f}")
    print(f"   Match threshold  : {threshold}")
    if sim >= threshold:
        print(f"   ✅ PASS — similarity ({sim:.4f}) >= threshold ({threshold})")
        print(f"      The Re-ID pipeline should recognise this person.")
    else:
        gap = threshold - sim
        print(f"   ❌ FAIL — similarity ({sim:.4f}) < threshold ({threshold})")
        print(f"      Gap: {gap:.4f} below threshold.")
        print(f"      Suggestions:")
        print(f"        - Use a screenshot from the actual video for registration")
        print(f"        - Increase --augmentations (e.g. --augmentations 15)")
        print(f"        - Add more photos of this person in different lighting")
        print(f"        - Check if the test image actually contains the same person")

    print()
    print("Per-embedding breakdown:")
    for i, emb in enumerate(record["embeddings"]):
        s = _cos(feat, emb)
        flag = "✅" if s >= threshold else "  "
        print(f"   {flag}  Photo {i+1}: {s:.4f}")
    print()


def _cosine_sim(a, b):
    from registration.autofix import cosine_sim
    return cosine_sim(a, b)


def _scan_video_for_candidates(video_path, detector, ref_avg, out_dir,
                               samples, augmentations):
    """Thin wrapper around registration.autofix.scan_video_for_candidates so
    the CLI and the web API share exactly the same scanning logic."""
    from registration.autofix import scan_video_for_candidates
    return scan_video_for_candidates(
        video_path, detector, ref_avg, out_dir, samples, augmentations)


def _print_candidates(top, threshold, out_dir):
    import os
    print(f"   {'#':>3s}  {'Video':>12s}  {'Frame':>5s}  {'Similarity':>10s}  {'Crop file':46s}")
    print("   " + "-" * 78)
    for i, (sim, vid_name, fid, x1, y1, w, h, path) in enumerate(top):
        label = f"  ✅" if sim >= threshold else ""
        print(f"   [{i}]  {vid_name:>12s}  {fid:>5d}  {sim:>8.4f}     "
              f"{os.path.basename(path):46s}{label}")
    print(f"   (crops saved to {out_dir}/)")


def cmd_autofix(args):
    """Auto-find a registered person across one or more videos, extract
    candidate crops, and let the user pick one for re-registration.

    Interactive mode (no --pick):
      - Scans video 1 and shows its top candidates.
      - If none of them is the right person, type `N` to move to video 2,
        then video 3, and so on. Type `Q` to quit, or a number to register
        that candidate.

    Scripted mode (--pick N):
      - Scans every video, ranks ALL candidates globally, and registers
        the N-th one (0-indexed). Used by the frontend gallery.
    """
    import os
    import numpy as np
    from registration.autofix import get_detector, get_autofix_dir
    from registration.identity_db import IdentityDatabase
    from registration.db_config import SEARCH_SETTINGS

    db = IdentityDatabase()
    if not db.person_exists(args.name):
        print(f"❌ '{args.name}' is not in the database.")
        return

    record = db.get_person(args.name)
    ref_avg = np.array(record["average_embedding"], dtype=np.float32)
    threshold = SEARCH_SETTINGS["match_threshold"]

    detector = get_detector()
    out_dir = get_autofix_dir()

    videos = args.video

    # ── Scripted mode: scan everything, pick by global rank ──────────────
    if args.pick is not None:
        all_cands = []
        for vid in videos:
            all_cands.extend(_scan_video_for_candidates(
                vid, detector, ref_avg, out_dir, args.samples, args.augmentations))
        all_cands.sort(key=lambda x: x[0], reverse=True)
        top = all_cands[:args.top]

        if not top:
            print("No persons detected across any of the videos.")
            return

        print(f"\nAuto-fix for '{args.name}' — top candidates across all videos:")
        _print_candidates(top, threshold, out_dir)

        if args.pick < 0 or args.pick >= len(top):
            print(f"❌ --pick {args.pick} out of range (0-{len(top)-1})")
            return

        sim, vid_name, fid, x1, y1, w, h, chosen_path = top[args.pick]
        print(f"\nRegistering '{args.name}' from candidate [{args.pick}] "
              f"(similarity {sim:.4f})...")
        from registration.register_person import register_person
        register_person(args.name, [chosen_path], overwrite=True,
                        num_augmentations=args.augmentations)
        print(f"✅ Re-registered '{args.name}' from {vid_name} frame {fid}")
        print(f"   Verify with: register.py verify --name '{args.name}' --test-image <another_crop>")
        return

    # ── Interactive mode: video by video ─────────────────────────────────
    print(f"\nAuto-fix for '{args.name}' — {len(videos)} video(s) available.")
    picked = None

    for vid_idx, vid in enumerate(videos):
        cands = _scan_video_for_candidates(
            vid, detector, ref_avg, out_dir, args.samples, args.augmentations)
        top = cands[:args.top]

        print(f"\n▶ Video {vid_idx + 1}/{len(videos)}: {vid}")
        if not top:
            print("   No persons detected in this video.")
            if vid_idx == len(videos) - 1:
                break
            print("   Press Enter to try the next video, or Q to quit.")
            if input("> ").strip().lower() == "q":
                break
            continue

        _print_candidates(top, threshold, out_dir)

        is_last = vid_idx == len(videos) - 1
        prompt = ("   Pick a candidate [0-4], N = next video, or Q to quit: "
                  if not is_last
                  else "   Pick a candidate [0-4], or Q to quit: ")
        choice = input(prompt).strip().lower()

        if choice == "q":
            break
        if choice == "n" and not is_last:
            continue
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(top):
                sim, vid_name, fid, x1, y1, w, h, chosen_path = top[idx]
                from registration.register_person import register_person
                register_person(args.name, [chosen_path], overwrite=True,
                                num_augmentations=args.augmentations)
                picked = (sim, vid_name, fid, chosen_path)
                break
            print(f"   ❌ {idx} is out of range (0-{len(top)-1})")
        else:
            print("   ❌ Invalid choice — enter a number, N, or Q.")

    if picked:
        sim, vid_name, fid, chosen_path = picked
        print(f"\n✅ Re-registered '{args.name}' from {vid_name} frame {fid} "
              f"(similarity {sim:.4f})")
        print(f"   Verify with: register.py verify --name '{args.name}' --test-image <another_crop>")
    else:
        print("\nNo candidate selected — registration unchanged.")



def main():
    parser = argparse.ArgumentParser(description="Person Registration & Identity Database")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Register one person from image files")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--images", nargs="+", required=True,
                        help="Image file paths or glob patterns")
    p_add.add_argument("--overwrite", action="store_true",
                        help="Replace this person's existing photos instead of adding to them")
    p_add.add_argument("--augmentations", type=int, default=5,
                        help="Number of augmented embeddings to average (default: 5; "
                             "0 = no random augmentation, just the preprocessed base image). "
                             "Higher values produce more domain-robust embeddings.")
    p_add.set_defaults(func=cmd_add)

    p_bulk = sub.add_parser("bulk", help="Register everyone under a folder-of-folders")
    p_bulk.add_argument("--dir", required=True,
                         help="Folder containing one sub-folder of images per person")
    p_bulk.add_argument("--augmentations", type=int, default=5,
                         help="Number of augmented embeddings per photo (default: 5)")
    p_bulk.set_defaults(func=cmd_bulk)

    p_list = sub.add_parser("list", help="List all registered persons")
    p_list.set_defaults(func=cmd_list)

    p_search = sub.add_parser("search", help="Search registered persons by (partial) name")
    p_search.add_argument("--name", required=True)
    p_search.set_defaults(func=cmd_search)

    p_verify = sub.add_parser("verify", help="Verify registration quality against a test image")
    p_verify.add_argument("--name", required=True,
                           help="Registered person's name to verify against")
    p_verify.add_argument("--test-image", required=True,
                           help="Path to a test image (e.g. a frame extracted from the video)")
    p_verify.add_argument("--augmentations", type=int, default=5,
                           help="Number of augmentations for the test image embedding (default: 5)")
    p_verify.set_defaults(func=cmd_verify)

    p_from = sub.add_parser("from-frame",
                             help="Register a person directly from a video frame crop")
    p_from.add_argument("--name", required=True,
                         help="Person's name")
    p_from.add_argument("--video", required=True,
                         help="Path to video file")
    p_from.add_argument("--frame", type=int, default=None,
                         help="Frame number (default: middle of video)")
    p_from.add_argument("--bbox", type=int, nargs=4, default=None,
                         help="Bounding box: x1 y1 x2 y2")
    p_from.add_argument("--detect", action="store_true",
                         help="Auto-detect persons in the frame instead of manual bbox")
    p_from.add_argument("--pick", type=int, default=None,
                         help="When used with --detect, register the N-th detected person (0-indexed)")
    p_from.add_argument("--overwrite", action="store_true",
                         help="Replace this person's existing photos")
    p_from.add_argument("--augmentations", type=int, default=5,
                         help="Number of augmentations (default: 5)")
    p_from.set_defaults(func=cmd_from_frame)

    p_autofix = sub.add_parser("auto-fix",
                                help="Auto-find a person in the video and suggest frame crops for re-registration")
    p_autofix.add_argument("--name", required=True,
                            help="Registered person's name")
    p_autofix.add_argument("--video", nargs="+", required=True,
                            help="One or more video file paths (scanned in order; press N to "
                                 "move to the next video if you don't like the current candidates)")
    p_autofix.add_argument("--samples", type=int, default=20,
                            help="Number of video frames to sample (default: 20)")
    p_autofix.add_argument("--top", type=int, default=5,
                            help="Number of top candidates to show (default: 5)")
    p_autofix.add_argument("--pick", type=int, default=None,
                            help="Register the N-th candidate (0-indexed) instead of just listing")
    p_autofix.add_argument("--augmentations", type=int, default=5,
                            help="Number of augmentations (default: 5)")
    p_autofix.set_defaults(func=cmd_autofix)

    p_delete = sub.add_parser("delete", help="Remove a registered person")
    p_delete.add_argument("--name", required=True)
    p_delete.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p_delete.set_defaults(func=cmd_delete)

    p_backup = sub.add_parser("backup", help="Export the whole database to a JSON file")
    p_backup.add_argument("--path", required=True, help="Where to write the backup, e.g. backups/db_2026-07-27.json")
    p_backup.set_defaults(func=cmd_backup)

    p_restore = sub.add_parser("restore", help="Restore the database from a backup JSON file")
    p_restore.add_argument("--path", required=True)
    p_restore.add_argument("--replace", action="store_true",
                            help="Replace the current database entirely instead of merging")
    p_restore.set_defaults(func=cmd_restore)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    main()
