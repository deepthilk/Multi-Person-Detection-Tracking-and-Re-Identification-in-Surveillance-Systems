"""
Fine-tune the ResNet-50 Re-ID backbone on Market-1501.
========================================================

IMPORTANT — this trains the *exact* backbone class that reidentification/
reid_main.py already uses at inference time (`ResNetReIDBackbone`). It is
imported from there, not redefined here, so there is zero risk of an
architecture mismatch between what gets trained and what the live pipeline
loads.

Loss  : ID cross-entropy (label smoothing) + batch-hard triplet loss
        — the standard recipe for metric-learning Re-ID (torchreid's
        ImageTripletEngine uses the same combination; we reimplement it
        directly in pure PyTorch because reid_main.py deliberately avoids
        importing torchreid — see the comment in ReIDEngine._load_model
        about the TensorFlow dependency chain causing hangs).
Sampler: P identities x K images per batch (PKSampler in dataset.py) so
        every batch has the positive/negative pairs triplet loss needs.

Output: reidentification/weights/best_model.pth — contains ONLY
        `ResNetReIDBackbone.state_dict()`, so reid_main.py's `_load_model`
        can load it with a single `load_state_dict()` call and nothing
        else in the pipeline has to change.

Usage:
    python -m reidentification.training.train_reid \
        --data-root reidentification/dataset/Market-1501 \
        --epochs 60 --device cuda

    # quick smoke test on a few identities, no GPU required:
    python -m reidentification.training.train_reid --epochs 1 --batch-p 4 --batch-k 2 --device cpu
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

from reidentification.reid_main import ResNetReIDBackbone   # SAME class used live
from reidentification.training.dataset import Market1501Dataset, PKSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
#  Training-only wrapper: backbone (kept identical to inference) + ID head
# ─────────────────────────────────────────────────────────────────────────

class ReIDTrainNet(nn.Module):
    """Wraps the production ResNetReIDBackbone with a classifier head used
    only during training. The head is discarded when saving — reid_main.py
    only ever needs `backbone`."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = ResNetReIDBackbone()          # -> 512-d, L2-normalised
        self.classifier = nn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        emb = self.backbone(x)          # already L2-normalised
        logits = self.classifier(emb)
        return emb, logits


# ─────────────────────────────────────────────────────────────────────────
#  Batch-hard triplet loss
# ─────────────────────────────────────────────────────────────────────────

def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3):
    """For every anchor in the batch: hardest positive (furthest same-id) and
    hardest negative (closest different-id), then hinge loss on the margin.
    Standard formulation used for Re-ID metric learning (Hermans et al. 2017)."""
    dist = torch.cdist(embeddings, embeddings, p=2)                 # [B, B]
    same = labels.unsqueeze(0) == labels.unsqueeze(1)                # [B, B]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    # hardest positive: max distance among same-identity pairs
    hardest_pos = (dist * pos_mask.float()).max(dim=1).values
    # hardest negative: min distance among different-identity pairs
    dist_neg = dist.clone()
    dist_neg[~neg_mask] = float("inf")
    hardest_neg = dist_neg.min(dim=1).values

    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────
#  Train / eval loop
# ─────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        logger.warning("CUDA requested but unavailable — falling back to CPU")

    train_dir = Path(args.data_root) / "bounding_box_train"
    if not train_dir.exists():
        logger.error(f"❌ Training images not found at {train_dir}. "
                      f"Download Market-1501 and point --data-root at it "
                      f"(the folder must contain bounding_box_train/, query/, bounding_box_test/).")
        return 1

    dataset = Market1501Dataset(str(train_dir))
    if len(dataset) == 0 or dataset.num_classes == 0:
        logger.error(f"❌ No usable images found under {train_dir}")
        return 1
    logger.info(f"Train images: {len(dataset)}  |  identities: {dataset.num_classes}")

    sampler = PKSampler(dataset, p=args.batch_p, k=args.batch_k)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = ReIDTrainNet(num_classes=dataset.num_classes).to(device)

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = out_path.with_name("last_checkpoint.pth")

    best_metric = -1.0
    query_dir = Path(args.data_root) / "query"
    gallery_dir = Path(args.data_root) / "bounding_box_test"
    can_validate = query_dir.exists() and gallery_dir.exists()
    if not can_validate:
        logger.warning("query/ or bounding_box_test/ not found — skipping rank-1/mAP "
                        "validation, saving the checkpoint from the final epoch instead. "
                        "For a real accuracy number, run evaluate.py once you have the full dataset.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_id, running_tri, n_batches = 0.0, 0.0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            embeddings, logits = model(imgs)

            id_loss = ce_loss(logits, labels)
            tri_loss = batch_hard_triplet_loss(embeddings, labels, margin=args.triplet_margin)
            loss = id_loss + tri_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_id += id_loss.item()
            running_tri += tri_loss.item()
            n_batches += 1

        scheduler.step()
        dt = time.time() - t0
        logger.info(f"Epoch {epoch:3d}/{args.epochs} | id_loss={running_id/max(1,n_batches):.4f} "
                    f"triplet_loss={running_tri/max(1,n_batches):.4f} | lr={scheduler.get_last_lr()[0]:.2e} "
                    f"| {dt:.1f}s")

        torch.save({"model_state": model.state_dict(), "epoch": epoch}, resume_path)

        do_eval = can_validate and (epoch % args.eval_every == 0 or epoch == args.epochs)
        if do_eval:
            from reidentification.training.evaluate import evaluate_model
            rank1, mAP = evaluate_model(model.backbone, str(query_dir), str(gallery_dir), device)
            logger.info(f"  ↳ val: rank-1={rank1:.4f}  mAP={mAP:.4f}")
            metric = mAP
            if metric > best_metric:
                best_metric = metric
                torch.save(model.backbone.state_dict(), out_path)
                logger.info(f"  ↳ ✅ new best (mAP={mAP:.4f}) -> {out_path}")
        elif not can_validate and epoch == args.epochs:
            # No ground truth to validate against — save the final epoch so
            # there's still something for reid_main.py to load.
            torch.save(model.backbone.state_dict(), out_path)
            logger.info(f"  ↳ saved final-epoch weights (no validation set available) -> {out_path}")

    logger.info("Training complete.")
    if can_validate:
        logger.info(f"Best mAP: {best_metric:.4f}  |  weights: {out_path}")
    return 0


def build_argparser():
    p = argparse.ArgumentParser(description="Fine-tune ResNetReIDBackbone on Market-1501")
    p.add_argument("--data-root", type=str, default="reidentification/dataset/Market-1501",
                    help="Folder containing bounding_box_train/, query/, bounding_box_test/")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-p", type=int, default=16, help="identities per batch")
    p.add_argument("--batch-k", type=int, default=4, help="images per identity per batch")
    p.add_argument("--lr", type=float, default=3.5e-4)
    p.add_argument("--step-size", type=int, default=20, help="StepLR decay every N epochs")
    p.add_argument("--triplet-margin", type=float, default=0.3)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--output", type=str, default="reidentification/weights/best_model.pth")
    p.add_argument("--resume", type=str, default=None, help="path to last_checkpoint.pth to resume from")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(train(args))
