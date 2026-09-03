"""Stream 4: frozen pretrained CLIP image encoder for a semantic embedding, injecting
"world knowledge" the from-scratch streams can't learn from ~4800 images alone.

We precompute CLIP embeddings once per image (frozen, no grad) and cache them to disk,
since re-running CLIP forward passes every epoch would dominate training time for no benefit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_EMBED_DIM = 512  # ViT-B/32 projection dim


class ClipEncoder:
    """Wraps a frozen HF CLIP model for batched image -> embedding extraction (no grad)."""

    def __init__(self, device: torch.device, model_name: str = CLIP_MODEL_NAME):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_pil_batch(self, pil_images: list) -> np.ndarray:
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


class ClipBranch(nn.Module):
    """Learned projection on top of precomputed, frozen CLIP embeddings. Input: (B, CLIP_EMBED_DIM)
    -> (B, out_dim)."""

    def __init__(self, in_dim: int = CLIP_EMBED_DIM, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):  # x: (B, CLIP_EMBED_DIM) precomputed embedding
        return self.proj(x)
