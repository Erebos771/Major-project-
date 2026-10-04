"""Stream 2: small CNN over the vegetation/color-index channel stack (see
src/data/vegetation_features.py). Input: (B, VEG_CHANNELS, 224, 224) -> (B, 64) feature vector.
"""
from __future__ import annotations

import torch.nn as nn

from src.data.vegetation_features import VEG_CHANNELS

OUT_DIM = 64


class VegColorBranch(nn.Module):
    def __init__(self, in_channels: int = VEG_CHANNELS, out_dim: int = OUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56
            nn.Conv2d(64, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # x: (B, VEG_CHANNELS, 224, 224)
        x = self.net(x)
        x = x.mean(dim=(2, 3))  # GAP -> (B, out_dim)
        return x
