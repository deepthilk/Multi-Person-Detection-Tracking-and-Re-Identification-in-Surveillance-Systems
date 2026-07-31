"""
Auto-fix candidate scanning — shared by the CLI (register.py auto-fix) and
the web API (web/registration_api.py).

Scans one or more videos, detects persons, embeds each crop, and compares
it against a registered person's embedding. Candidates are saved as JPEG
crops (filename includes the video name so runs never overwrite each other)
and returned sorted by similarity.
"""

import os
from pathlib import Path

import numpy as np

from registration.embedder import embed_image


def cosine_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def scan_video_for_candidates(video_path, detector, ref_avg, out_dir,
                              samples=20, augmentations=5) -> list:
    """Sample `video_path`, detect persons, embed each crop, and compare it
    against the registered embedding.

    Returns a list of tuples sorted by similarity (highest first):
        (sim, video_name, frame_id, x1, y1, w, h, crop_path)
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_name = Path(video_path).stem
    frame_indices = [int(total * i / (samples + 1)) for i in range(1, samples + 1)]

    candidates = []
    seen = set()

    for fid in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        dets = detector.detect(frame)
        for d in dets:
            x1, y1, w, h, conf = d
            crop = frame[y1:y1 + h, x1:x1 + w]
            if crop.size == 0:
                continue
            key = (fid, x1, y1, w, h)
            if key in seen:
                continue
            seen.add(key)

            crop_path = str(out_dir / f"candidate_{vid_name}_{fid}_{x1}_{y1}.jpg")
            cv2.imwrite(crop_path, crop)

            feat = embed_image(crop_path, num_augmentations=augmentations)
            if feat is None:
                continue
            sim = cosine_sim(feat, ref_avg)
            candidates.append((sim, vid_name, fid, x1, y1, w, h, crop_path))

    cap.release()
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def get_detector(conf_threshold: float = 0.25):
    """Lazily build the person detector (used by CLI + API alike)."""
    from detection.detect_module import PersonDetector
    return PersonDetector(conf_threshold=conf_threshold)


def get_autofix_dir() -> Path:
    out_dir = Path("outputs/registration/_auto_fix")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
