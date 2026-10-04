"""Build the paper-parity balanced subset from the raw Kaggle
"Weed Detection in Soybean Crops" dataset.

Expects raw data at DATA_RAW_DIR either as:
  - data/raw/weed-detection-in-soybean-crops.zip  (unzipped automatically), or
  - data/raw/<already unzipped, one subfolder per class>

Class folder names in the Kaggle dataset are: broadleaf, grass, soil, soybean
(case-insensitive match is attempted; adjust CLASS_DIR_ALIASES below if the
actual archive uses different folder names).

Output: data/processed/{train,test}/<class>/*.jpg  (224x224 RGB, JPEG)
        data/processed/manifest.csv  (path, class, split)

Usage:
    python -m src.data.prepare_dataset
"""
from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

RAW_DIR = Path("dataset/dataset")
PROCESSED_DIR = Path("data/processed")
IMG_SIZE = 224
SEED = 42
TEST_FRAC = 0.20

CLASSES = ["broadleaf", "grass", "soil", "soybean"]
MAX_PER_CLASS = {"broadleaf": 1195, "grass": 1200, "soil": 1200, "soybean": 1200}

# If the unzipped folder names differ from CLASSES, map them here (lowercased match).
CLASS_DIR_ALIASES = {
    "broadleaf": ["broadleaf", "broad_leaf", "broad-leaf"],
    "grass": ["grass"],
    "soil": ["soil"],
    "soybean": ["soybean", "soy"],
}

IMAGE_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


def _maybe_unzip() -> None:
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        return
    for z in zips:
        marker = RAW_DIR / f".unzipped_{z.stem}"
        if marker.exists():
            continue
        print(f"Unzipping {z} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
        marker.touch()


def _find_class_dir(cls: str) -> Path:
    aliases = {a.lower() for a in CLASS_DIR_ALIASES[cls]}
    candidates = [p for p in RAW_DIR.rglob("*") if p.is_dir() and p.name.lower() in aliases]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a folder for class '{cls}' under {RAW_DIR}. "
            f"Looked for names: {sorted(aliases)}. "
            f"Adjust CLASS_DIR_ALIASES in src/data/prepare_dataset.py if needed."
        )
    # Prefer the shallowest match, and the one with the most images if there's a tie.
    candidates.sort(key=lambda p: (len(p.parts), -sum(1 for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)))
    return candidates[0]


def _systematic_sample(files: list[Path], k: int) -> list[Path]:
    """Systematic traversal sample: evenly spaced picks across the sorted file list."""
    files = sorted(files)
    n = len(files)
    if k >= n:
        return files
    step = n / k
    return [files[int(i * step)] for i in range(k)]


def build_manifest() -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for cls in CLASSES:
        class_dir = _find_class_dir(cls)
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        cap = MAX_PER_CLASS[cls]
        if len(files) < cap:
            print(f"WARNING: class '{cls}' has only {len(files)} images, expected >= {cap}")
        chosen = _systematic_sample(files, min(cap, len(files)))
        print(f"{cls}: {len(chosen)} images selected (from {len(files)} available) @ {class_dir}")
        items.extend((f, cls) for f in chosen)
    return items


def convert_and_resize(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        im.save(dst, "JPEG", quality=95)


def main() -> None:
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        raise SystemExit(
            f"{RAW_DIR} is empty. Download the dataset from "
            "https://www.kaggle.com/datasets/fpeccia/weed-detection-in-soybean-crops "
            f"and place the zip (or unzipped folders) under {RAW_DIR}/."
        )
    _maybe_unzip()

    manifest = build_manifest()
    paths = [p for p, _ in manifest]
    labels = [c for _, c in manifest]

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=TEST_FRAC, random_state=SEED, stratify=labels
    )

    rows = []
    for split, split_paths, split_labels in [
        ("train", train_paths, train_labels),
        ("test", test_paths, test_labels),
    ]:
        for src, cls in zip(split_paths, split_labels):
            dst = PROCESSED_DIR / split / cls / f"{src.stem}.jpg"
            convert_and_resize(src, dst)
            rows.append((str(dst), cls, split))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "split"])
        w.writerows(rows)

    print("\n--- Summary ---")
    print(f"Total: {len(rows)}  (expected 4795)")
    print(f"Train: {len(train_paths)}  Test: {len(test_paths)}  (expected 3836 / 959)")
    for split, split_labels in [("train", train_labels), ("test", test_labels)]:
        counts = {c: split_labels.count(c) for c in CLASSES}
        print(f"{split}: {counts}")


if __name__ == "__main__":
    main()
