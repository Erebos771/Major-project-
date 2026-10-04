"""Variant 2: lighter branches. Depthwise-separable convs + GELU for veg/texture, 1-block ViT,
and the plain concat fusion head from the main model."""
from __future__ import annotations

import torch.nn as nn

from src.data.texture_features import TEXTURE_CHANNELS
from src.data.vegetation_features import VEG_CHANNELS
from src.models.fusion import MultiStreamModel
from src.models.streams.clip_semantic import ClipBranch
from src.models.streams.vit_cab import CABBranch, ViTBranch


def _sep_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, padding=1, groups=cin),
        nn.Conv2d(cin, cout, 1),
        nn.BatchNorm2d(cout),
        nn.GELU(),
    )


class SepConvBranch(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            _sep_block(in_channels, 32), nn.MaxPool2d(2),
            _sep_block(32, 64), nn.MaxPool2d(2),
            _sep_block(64, out_dim),
        )

    def forward(self, x):
        return self.net(x).mean(dim=(2, 3))


def build_light_branches(streams: list[str]) -> dict:
    b = {}
    if "rgb" in streams:
        b["rgb_vit"] = ViTBranch(num_blocks=1)
        b["rgb_cab"] = CABBranch()
    if "veg" in streams:
        b["veg"] = SepConvBranch(VEG_CHANNELS)
    if "texture" in streams:
        b["texture"] = SepConvBranch(TEXTURE_CHANNELS)
    if "clip" in streams:
        b["clip"] = ClipBranch()
    return b


def build_model(streams, num_classes=4):
    return MultiStreamModel(build_light_branches(streams), num_classes=num_classes, fusion="concat")
