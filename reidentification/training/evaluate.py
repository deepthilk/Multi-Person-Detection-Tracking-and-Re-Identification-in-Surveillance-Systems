"""
Market-1501 evaluation: CMC (rank-1/5/10) and mAP.
=====================================================

Standard single-query protocol: for every query image, gallery images that
share BOTH the same person id AND the same camera id are excluded (they're
the same physical sighting, not a valid re-identification target), along
with junk detections (pid == -1). What's left is ranked by embedding
distance and scored.

Usage:
    python -m reidentification.training.evaluate \
        --weights reidentification/weights/best_model.pth \
        --data-root reidentification/dataset/Market-1501 --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reidentification.reid_main import ResNetReIDBackbone
from reidentification.training.dataset import Market1501EvalDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def _extract_embeddings(model, root_dir, device, batch_size=64, workers=4):
    ds = Market1501EvalDataset(root_dir)
    if len(ds) == 0:
        return np.zeros((0, 512), dtype=np.float32), np.array([]), np.array([])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    model.eval()
    feats, pids, camids = [], [], []
    for imgs, p, c in loader:
        imgs = imgs.to(device)
        emb = model(imgs)
        feats.append(emb.cpu().numpy())
        pids.append(p.numpy())
        camids.append(c.numpy())
    return np.concatenate(feats), np.concatenate(pids), np.concatenate(camids)


def _compute_cmc_map(dist, q_pids, q_camids, g_pids, g_camids, max_rank=10):
    """dist: [num_q, num_g] distance matrix (lower = more similar)."""
    num_q, num_g = dist.shape
    indices = np.argsort(dist, axis=1)

    all_cmc = []
    all_ap = []
    valid_queries = 0

    for i in range(num_q):
        order = indices[i]
        q_pid, q_cam = q_pids[i], q_camids[i]

        # junk mask: same pid+cam (same physical sighting) OR pid == -1
        g_pid_ord = g_pids[order]
        g_cam_ord = g_camids[order]
        remove = ((g_pid_ord == q_pid) & (g_cam_ord == q_cam)) | (g_pid_ord == -1)
        keep = ~remove

        matches = (g_pid_ord[keep] == q_pid).astype(np.int32)
        if matches.sum() == 0:
            continue  # this query has no valid gallery match — skip (standard protocol)
        valid_queries += 1

        # CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        # AP
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        precision_at_k = tmp_cmc / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / num_rel
        all_ap.append(ap)

    if valid_queries == 0:
        logger.warning("No valid queries after filtering — check that query/ and "
                        "bounding_box_test/ belong to the same dataset split.")
        return np.zeros(max_rank), 0.0

    # pad CMC vectors that are shorter than max_rank (few gallery candidates)
    padded = np.zeros((len(all_cmc), max_rank))
    for i, c in enumerate(all_cmc):
        padded[i, :len(c)] = c
        if len(c) < max_rank:
            padded[i, len(c):] = c[-1]  # once matched, stays matched
    cmc = padded.mean(axis=0)
    mAP = float(np.mean(all_ap))
    return cmc, mAP


def evaluate_model(model, query_dir, gallery_dir, device, batch_size=64, workers=4):
    """Returns (rank1, mAP). `model` must already be on `device` and output
    L2-normalised embeddings (i.e. it's a ResNetReIDBackbone instance)."""
    q_feat, q_pids, q_camids = _extract_embeddings(model, query_dir, device, batch_size, workers)
    g_feat, g_pids, g_camids = _extract_embeddings(model, gallery_dir, device, batch_size, workers)

    if len(q_feat) == 0 or len(g_feat) == 0:
        logger.warning("Empty query or gallery set — cannot evaluate.")
        return 0.0, 0.0

    # embeddings are L2-normalised -> cosine distance = 1 - dot product
    dist = 1.0 - q_feat @ g_feat.T
    cmc, mAP = _compute_cmc_map(dist, q_pids, q_camids, g_pids, g_camids)
    return float(cmc[0]), mAP


def main():
    p = argparse.ArgumentParser(description="Evaluate a Re-ID backbone on Market-1501")
    p.add_argument("--weights", type=str, default="reidentification/weights/best_model.pth")
    p.add_argument("--data-root", type=str, default="reidentification/dataset/Market-1501")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = ResNetReIDBackbone().to(device)
    weights_path = Path(args.weights)
    if weights_path.exists() and weights_path.stat().st_size > 0:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        logger.info(f"Loaded fine-tuned weights from {weights_path}")
    else:
        logger.warning(f"{weights_path} not found/empty — evaluating the ImageNet-only "
                        f"backbone (expect poor Re-ID accuracy, this is just a sanity baseline).")

    query_dir = str(Path(args.data_root) / "query")
    gallery_dir = str(Path(args.data_root) / "bounding_box_test")

    rank1, mAP = evaluate_model(model, query_dir, gallery_dir, device, batch_size=args.batch_size)
    logger.info("=" * 50)
    logger.info(f"Rank-1 : {rank1:.4f}")
    logger.info(f"mAP    : {mAP:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
