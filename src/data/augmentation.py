"""Heavier data augmentation for the "additional improvements" pass: rotation, color jitter,
random resized crop, plus batch-level CutMix/MixUp. The grass<->broadleaf/soybean confusion
looks partly like a generalization gap, not just a representation gap, so this targets that.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms

HEAVY_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
    ]
)


def mixup(images: torch.Tensor, labels: torch.Tensor, num_classes: int, alpha: float = 0.2):
    """images: (B,C,H,W) float. labels: (B,) long. Returns (mixed_images, mixed_onehot_labels)."""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[perm]
    onehot = torch.nn.functional.one_hot(labels, num_classes).float()
    mixed_labels = lam * onehot + (1 - lam) * onehot[perm]
    return mixed, mixed_labels


def cutmix(images: torch.Tensor, labels: torch.Tensor, num_classes: int, alpha: float = 1.0):
    """images: (B,C,H,W) float. labels: (B,) long. Returns (mixed_images, mixed_onehot_labels)."""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0), device=images.device)
    B, C, H, W = images.shape

    r = np.sqrt(1 - lam)
    cut_h, cut_w = int(H * r), int(W * r)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
    lam_adjusted = 1 - ((y2 - y1) * (x2 - x1) / (H * W))

    onehot = torch.nn.functional.one_hot(labels, num_classes).float()
    mixed_labels = lam_adjusted * onehot + (1 - lam_adjusted) * onehot[perm]
    return mixed, mixed_labels


def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against soft (mixup/cutmix) targets."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()
