"""Dataset that serves RGB (on-the-fly) + precomputed veg/texture/clip features (from
src/data/precompute_features.py) for the multi-stream fusion model. Which streams are
returned is configurable so we can run the incremental ablation
(RGB-only -> +veg -> +texture -> +CLIP) without rewriting the loader each time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.dataset import read_manifest

FEATURES_DIR = Path("data/processed/features")


class MultiStreamDataset(Dataset):
    def __init__(self, split: str, streams: tuple[str, ...] = ("rgb", "veg", "texture", "clip"), rgb_transform=None):
        self.split = split
        self.streams = streams
        self.items = read_manifest(split)
        self.rgb_transform = rgb_transform

        feat_dir = FEATURES_DIR / split
        self.labels = np.load(feat_dir / "labels.npy")
        assert len(self.labels) == len(self.items), (
            f"labels.npy ({len(self.labels)}) doesn't match manifest ({len(self.items)}) for split={split}. "
            "Re-run precompute_features.py if the manifest changed."
        )

        self.cache = {}
        if "veg" in streams:
            self.cache["veg"] = np.load(feat_dir / "veg.npy", mmap_mode="r")
        if "texture" in streams:
            self.cache["texture"] = np.load(feat_dir / "texture.npy", mmap_mode="r")
        if "clip" in streams:
            self.cache["clip"] = np.load(feat_dir / "clip.npy", mmap_mode="r")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, cls = self.items[idx]
        out = {}
        if "rgb" in self.streams:
            img = Image.open(path).convert("RGB")
            if self.rgb_transform is not None:
                img = self.rgb_transform(img)
            else:
                img = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            out["rgb"] = img
        if "veg" in self.streams:
            out["veg"] = torch.from_numpy(np.asarray(self.cache["veg"][idx]))
        if "texture" in self.streams:
            out["texture"] = torch.from_numpy(np.asarray(self.cache["texture"][idx]))
        if "clip" in self.streams:
            out["clip"] = torch.from_numpy(np.asarray(self.cache["clip"][idx]))
        label = int(self.labels[idx])
        return out, label


def collate_multistream(batch):
    inputs, labels = zip(*batch)
    keys = inputs[0].keys()
    collated = {k: torch.stack([x[k] for x in inputs]) for k in keys}
    labels = torch.tensor(labels, dtype=torch.long)
    return collated, labels
