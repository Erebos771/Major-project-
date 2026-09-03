"""Grad-CAM visualization for the RGB stream(s).

For HybridViTCAB (Stream 1), we target the CAB branch's last conv layer — it has a clean
spatial feature map before GAP, unlike the ViT branch (whose "spatial" signal lives in
attention weights over patch tokens, not a conv feature map, so classic Grad-CAM doesn't
apply directly there). This mirrors what the paper does for explainability comparison.

Usage:
    python -m src.eval.gradcam --ckpt checkpoints/stream1_vit_cab_best.pt --n 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from src.data.dataset import CLASSES, read_manifest
from src.models.streams.vit_cab import HybridViTCAB

OUT_DIR = Path("results/gradcam")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/stream1_vit_cab_best.pt")
    ap.add_argument("--n", type=int, default=8, help="images per class")
    ap.add_argument("--tag", default="stream1")
    args = ap.parse_args()

    device = get_device()
    model = HybridViTCAB(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    target_layer = model.cab.conv2  # last conv before SE+GAP in the CAB branch
    cam = GradCAM(model=model, target_layers=[target_layer])

    tf = transforms.ToTensor()
    test_items = read_manifest("test")
    by_class = {c: [] for c in CLASSES}
    for path, cls in test_items:
        by_class[cls].append(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cls, paths in by_class.items():
        for path in paths[: args.n]:
            img = Image.open(path).convert("RGB")
            rgb_float = np.asarray(img, dtype=np.float32) / 255.0
            tensor = tf(img).unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=tensor)[0]
            vis = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
            out_path = OUT_DIR / f"{args.tag}_{cls}_{Path(path).stem}.png"
            Image.fromarray(vis).save(out_path)
    print(f"Saved Grad-CAM visualizations to {OUT_DIR}/")


if __name__ == "__main__":
    main()
