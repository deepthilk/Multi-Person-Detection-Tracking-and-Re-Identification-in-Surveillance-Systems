"""
Market-1501 datasets for training and evaluation.
====================================================

Filename convention used by Market-1501 (and respected by this file):

    0001_c1s1_001051_00.jpg
    ^^^^ ^^         person id (pid).  "-1" = junk/distractor, "0000" = distractor
         ^^         camera id (c1..c6)

- Market1501Dataset        -> for TRAINING on bounding_box_train/
                               (junk pid==-1 dropped, pids remapped to a
                               contiguous 0..N-1 range for the classifier head)
- Market1501EvalDataset    -> for EVALUATION on query/ and bounding_box_test/
                               (junk kept out of results but camera id is kept
                               so evaluate.py can apply the same-camera/
                               same-identity exclusion rule)
"""

import os
import re
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

_FNAME_RE = re.compile(r"^(-?\d+)_c(\d+)s(\d+)_")


def _parse_fname(fname):
    """Return (pid, camera_id) parsed from a Market-1501 filename, or None
    if the filename doesn't match the expected pattern."""
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    pid = int(m.group(1))
    cam = int(m.group(2))
    return pid, cam


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.20)),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


class Market1501Dataset(Dataset):
    """Training split. Junk images (pid == -1) are dropped. Person IDs are
    remapped to a contiguous [0, num_classes) range because nn.CrossEntropyLoss
    needs dense class indices — Market-1501 pids themselves have gaps."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or TRAIN_TRANSFORM

        self.image_paths = []
        self.raw_pids = []
        self.cam_ids = []

        for file in sorted(os.listdir(root_dir)):
            if not file.endswith(".jpg"):
                continue
            parsed = _parse_fname(file)
            if parsed is None:
                continue
            pid, cam = parsed
            if pid == -1:          # junk, not a real identity
                continue
            self.image_paths.append(os.path.join(root_dir, file))
            self.raw_pids.append(pid)
            self.cam_ids.append(cam)

        # Contiguous label remap: {raw_pid: 0..N-1}, sorted for reproducibility
        unique_pids = sorted(set(self.raw_pids))
        self.pid2label = {pid: i for i, pid in enumerate(unique_pids)}
        self.labels = [self.pid2label[p] for p in self.raw_pids]
        self.num_classes = len(unique_pids)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

    @property
    def pids_per_image(self):
        """Raw (non-remapped) pid for every image — used by the balanced
        P x K batch sampler so it can group images by identity."""
        return self.raw_pids


class Market1501EvalDataset(Dataset):
    """Query / gallery split for evaluation. Keeps the ORIGINAL Market-1501
    pid (not remapped) plus camera id, since CMC/mAP need both to apply the
    same-camera-same-identity exclusion rule."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or EVAL_TRANSFORM

        self.image_paths = []
        self.pids = []
        self.cam_ids = []

        for file in sorted(os.listdir(root_dir)):
            if not file.endswith(".jpg"):
                continue
            parsed = _parse_fname(file)
            if parsed is None:
                continue
            pid, cam = parsed
            # Keep junk (-1) and distractors (0) OUT of query, but they are
            # legitimately part of the gallery in the official protocol —
            # evaluate.py filters them per-query via the junk mask instead
            # of dropping them here, so both splits can reuse this class.
            self.image_paths.append(os.path.join(root_dir, file))
            self.pids.append(pid)
            self.cam_ids.append(cam)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image, self.pids[idx], self.cam_ids[idx]


class PKSampler(torch.utils.data.Sampler):
    """Batch sampler for triplet/metric learning: each batch contains P
    identities x K images per identity (batch size = P*K). Falls back to
    sampling with replacement for identities that have fewer than K images."""

    def __init__(self, dataset: Market1501Dataset, p: int = 16, k: int = 4):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.index_by_label = {}
        for idx, label in enumerate(dataset.labels):
            self.index_by_label.setdefault(label, []).append(idx)
        self.labels = list(self.index_by_label.keys())

    def __iter__(self):
        import random
        labels = self.labels.copy()
        random.shuffle(labels)
        batch = []
        for label in labels:
            pool = self.index_by_label[label]
            chosen = (random.sample(pool, self.k) if len(pool) >= self.k
                      else [random.choice(pool) for _ in range(self.k)])
            batch.extend(chosen)
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return len(self.labels) // self.p


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "reidentification/dataset/Market-1501/bounding_box_train"
    dataset = Market1501Dataset(root)
    print("Total training images:", len(dataset))
    print("Total identities     :", dataset.num_classes)
    if len(dataset):
        image, label = dataset[0]
        print("Image shape:", image.shape)
        print("Label      :", label)
