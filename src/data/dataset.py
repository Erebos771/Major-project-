"""PyTorch Dataset for the processed weed classification images."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["broadleaf", "grass", "soil", "soybean"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

MANIFEST_PATH = Path("data/processed/manifest.csv")


def read_manifest(split: str, manifest_path: Path = MANIFEST_PATH) -> list[tuple[str, str]]:
    rows = []
    with open(manifest_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                rows.append((r["path"], r["class"]))
    return rows


class WeedDataset(Dataset):
    """Returns (image_float32_HWC_[0,1], label_idx). Transform hook can override output shape."""

    def __init__(self, split: str, transform=None, manifest_path: Path = MANIFEST_PATH):
        self.items = read_manifest(split, manifest_path)
        self.transform = transform
        if not self.items:
            raise RuntimeError(
                f"No items found for split='{split}' in {manifest_path}. "
                "Run `python -m src.data.prepare_dataset` first."
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, cls = self.items[idx]
        img = Image.open(path).convert("RGB")
        label = CLASS_TO_IDX[cls]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.asarray(img, dtype=np.float32) / 255.0
        return img, label
