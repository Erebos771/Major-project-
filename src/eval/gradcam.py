"""Grad-CAM visualization for the RGB stream(s).

For HybridViTCAB (Stream 1) and the full multi-stream model, we target the RGB CAB branch's
last conv layer — it has a clean spatial feature map before GAP, unlike the ViT branch (whose
"spatial" signal lives in attention weights over patch tokens, not a conv feature map, so
classic Grad-CAM doesn't apply directly there). This mirrors what the paper does for
explainability comparison.

Usage:
    python -m src.eval.gradcam --model stream1 --ckpt checkpoints/stream1_vit_cab_best.pt --n 8
    python -m src.eval.gradcam --model full --ckpt checkpoints/full_attention_best.pt \
        --streams rgb,veg,texture,clip --fusion attention --n 8 --tag final_model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from src.data.dataset import CLASSES, read_manifest
from src.data.texture_features import compute_texture_stack
from src.data.vegetation_features import compute_vegetation_stack
from src.models.fusion import MultiStreamModel
from src.models.streams.vit_cab import HybridViTCAB
from src.train.train_multistream import build_branches

OUT_DIR = Path("results/gradcam")
TEXTURE_RES = 56


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FixedAuxWrapper(nn.Module):
    """Wraps MultiStreamModel so Grad-CAM (which only knows how to vary one input tensor)
    can drive it: takes the RGB tensor as the sole forward() argument, and injects the other
    streams' tensors (computed once per image, held fixed) internally."""

    def __init__(self, model: MultiStreamModel, aux_inputs: dict):
        super().__init__()
        self.model = model
        self.aux_inputs = aux_inputs

    def forward(self, rgb_tensor):
        inputs = dict(self.aux_inputs)
        inputs["rgb_vit"] = rgb_tensor
        inputs["rgb_cab"] = rgb_tensor
        return self.model(inputs)


def build_full_model_and_cam(ckpt_path: str, streams: list[str], fusion: str, device: torch.device):
    from src.models.streams.clip_semantic import ClipEncoder

    branches = build_branches(streams)
    model = MultiStreamModel(branches, num_classes=len(CLASSES), fusion=fusion)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    clip_encoder = ClipEncoder(device) if "clip" in streams else None
    return model, clip_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["stream1", "full"], default="stream1")
    ap.add_argument("--ckpt", default="checkpoints/stream1_vit_cab_best.pt")
    ap.add_argument("--streams", default="rgb,veg,texture,clip", help="only used when --model full")
    ap.add_argument("--fusion", default="attention", help="only used when --model full")
    ap.add_argument("--n", type=int, default=8, help="images per class")
    ap.add_argument("--tag", default="stream1")
    args = ap.parse_args()

    device = get_device()
    tf = transforms.ToTensor()

    if args.model == "stream1":
        model = HybridViTCAB(num_classes=len(CLASSES))
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.to(device).eval()
        target_layer = model.cab.conv2
        cam_model = model
        clip_encoder = None
        streams = ["rgb"]
    else:
        streams = args.streams.split(",")
        full_model, clip_encoder = build_full_model_and_cam(args.ckpt, streams, args.fusion, device)
        target_layer = full_model.branches["rgb_cab"].conv2
        cam_model = None  # built per-image below since aux inputs vary per image

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

            if args.model == "stream1":
                cam = GradCAM(model=cam_model, target_layers=[target_layer])
            else:
                aux = {}
                if "veg" in streams:
                    veg = compute_vegetation_stack(rgb_float).transpose(2, 0, 1)
                    aux["veg"] = torch.from_numpy(veg).unsqueeze(0).to(device)
                if "texture" in streams:
                    import cv2

                    tex = compute_texture_stack(rgb_float)
                    tex = cv2.resize(tex, (TEXTURE_RES, TEXTURE_RES), interpolation=cv2.INTER_AREA)
                    tex = tex.transpose(2, 0, 1).astype(np.float32)
                    aux["texture"] = torch.from_numpy(tex).unsqueeze(0).to(device)
                if "clip" in streams:
                    emb = clip_encoder.encode_pil_batch([img])
                    aux["clip"] = torch.from_numpy(emb).to(device)
                wrapped = FixedAuxWrapper(full_model, aux).to(device).eval()
                cam = GradCAM(model=wrapped, target_layers=[target_layer])

            grayscale_cam = cam(input_tensor=tensor, targets=None)[0]
            vis = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
            out_path = OUT_DIR / f"{args.tag}_{cls}_{Path(path).stem}.png"
            Image.fromarray(vis).save(out_path)
    print(f"Saved Grad-CAM visualizations to {OUT_DIR}/")


if __name__ == "__main__":
    main()
