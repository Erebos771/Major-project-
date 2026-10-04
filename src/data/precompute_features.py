"""Precompute per-image feature caches for Streams 2-4 so training doesn't redo expensive
CPU work (sliding-window GLCM, CLIP forward passes) every epoch.

Writes, for each split (train/test):
  data/processed/features/<split>/veg.npy       (N, 9, 224, 224) float32
  data/processed/features/<split>/texture.npy   (N, 12, TEXTURE_RES, TEXTURE_RES) float16
  data/processed/features/<split>/clip.npy      (N, 512) float32
  data/processed/features/<split>/labels.npy    (N,) int64  (order matches manifest rows for that split)

Usage:
    python -m src.data.precompute_features [--streams veg,texture,clip] [--splits train,test]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.data.dataset import CLASS_TO_IDX, read_manifest
from src.data.texture_features import compute_texture_stack
from src.data.vegetation_features import compute_vegetation_stack

OUT_DIR = Path("data/processed/features")
TEXTURE_RES = 56  # texture branch downsamples 224->56 internally anyway; see note below


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def precompute_veg_texture(split: str, which: set[str]) -> None:
    items = read_manifest(split)
    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.array([CLASS_TO_IDX[c] for _, c in items], dtype=np.int64)
    np.save(out_dir / "labels.npy", labels)

    if "veg" in which:
        veg_path = out_dir / "veg.npy"
        if not veg_path.exists():
            veg_arr = np.zeros((len(items), 9, 224, 224), dtype=np.float32)
            for i, (path, _) in enumerate(tqdm(items, desc=f"[{split}] vegetation")):
                img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
                stack = compute_vegetation_stack(img)  # HWC
                veg_arr[i] = stack.transpose(2, 0, 1)
            np.save(veg_path, veg_arr)
        else:
            print(f"[{split}] veg.npy exists, skipping")

    if "texture" in which:
        tex_path = out_dir / "texture.npy"
        if not tex_path.exists():
            # Stored at TEXTURE_RES (not 224) and in float16: the texture branch downsamples
            # 224->56 internally via two maxpools anyway, and GLCM maps are already coarse
            # (computed on a 16x16 window grid then upsampled) — storing at full 224 float32
            # made the cache huge (9.2GB/split) and turned every epoch's shuffled reads into a
            # random-I/O bottleneck (~10 min/epoch instead of ~3). This cuts the cache ~30x.
            tex_arr = np.zeros((len(items), 12, TEXTURE_RES, TEXTURE_RES), dtype=np.float16)
            for i, (path, _) in enumerate(tqdm(items, desc=f"[{split}] texture")):
                img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
                stack = compute_texture_stack(img)  # HWC, 224x224
                stack = cv2.resize(stack, (TEXTURE_RES, TEXTURE_RES), interpolation=cv2.INTER_AREA)
                tex_arr[i] = stack.transpose(2, 0, 1).astype(np.float16)
            np.save(tex_path, tex_arr)
        else:
            print(f"[{split}] texture.npy exists, skipping")


def precompute_clip(split: str, batch_size: int = 64) -> None:
    from src.models.streams.clip_semantic import ClipEncoder

    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_path = out_dir / "clip.npy"
    if clip_path.exists():
        print(f"[{split}] clip.npy exists, skipping")
        return

    items = read_manifest(split)
    device = get_device()
    encoder = ClipEncoder(device)

    feats = []
    for i in tqdm(range(0, len(items), batch_size), desc=f"[{split}] CLIP"):
        batch_paths = [p for p, _ in items[i : i + batch_size]]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        feats.append(encoder.encode_pil_batch(imgs))
    feats = np.concatenate(feats, axis=0)
    np.save(clip_path, feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="veg,texture,clip")
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()
    which = set(args.streams.split(","))
    splits = args.splits.split(",")

    for split in splits:
        precompute_veg_texture(split, which & {"veg", "texture"})
        if "clip" in which:
            precompute_clip(split)


if __name__ == "__main__":
    main()
