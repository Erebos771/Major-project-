"""Stream 1: faithful reproduction of the paper's ViT + CAB hybrid (Eq 1-3, Fig 1).

ViT branch:
  Conv2D patch embedding (64 filters, 16x16 kernel, stride 16) -> 2 stacked transformer
  encoder blocks (4 heads, key_dim 128, FFN 128->64, residual + LayerNorm each) -> GAP -> 64-d

CAB branch (Convolutional Attention Block):
  2x [Conv2D 3x3, 64 filters, BatchNorm, ReLU] -> squeeze-excitation channel attention
  (GAP -> Dense(64, ReLU) -> Dense(64, sigmoid) -> channel-wise multiply) -> GAP -> 64-d

Fusion: concat(ViT 64, CAB 64) = 128 -> Dense(128) -> Dropout(0.4) -> Dense(num_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn

IMG_SIZE = 224
PATCH = 16
NUM_PATCHES = (IMG_SIZE // PATCH) ** 2  # 14*14 = 196
EMBED_DIM = 64
NUM_HEADS = 4
KEY_DIM = 128
FFN_HIDDEN = 128


class TransformerEncoderBlock(nn.Module):
    """Multi-head self-attention (key_dim=128 per head, as in the paper) + FFN(128->64), each with
    residual + LayerNorm, matching a Keras-style encoder block."""

    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, key_dim=KEY_DIM, ffn_hidden=FFN_HIDDEN):
        super().__init__()
        # Project to num_heads * key_dim for Q/K/V (mirrors Keras MultiHeadAttention(key_dim=128)),
        # then back down to embed_dim so the residual add is well-defined.
        inner_dim = num_heads * key_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, inner_dim)
        self.k_proj = nn.Linear(embed_dim, inner_dim)
        self.v_proj = nn.Linear(embed_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, embed_dim)
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_hidden, embed_dim),
        )

    def forward(self, x):  # x: (B, N, D)
        residual = x
        h = self.norm1(x)
        B, N, _ = h.shape
        q = self.q_proj(h).view(B, N, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, N, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.num_heads, self.key_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, self.num_heads * self.key_dim)
        attn = self.out_proj(attn)
        x = residual + attn

        residual = x
        h = self.norm2(x)
        h = self.ffn(h)
        x = residual + h
        return x


class ViTBranch(nn.Module):
    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, EMBED_DIM, kernel_size=PATCH, stride=PATCH)
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, EMBED_DIM))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([TransformerEncoderBlock() for _ in range(num_blocks)])

    def forward(self, x):  # x: (B, 3, 224, 224)
        x = self.patch_embed(x)  # (B, 64, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 64)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)  # GAP over patches -> (B, 64)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x):  # x: (B, C, H, W)
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))  # GAP -> (B, C)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class CABBranch(nn.Module):
    """Convolutional Attention Block: 2x Conv-BN-ReLU + SE channel attention -> GAP."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, out_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.se = SqueezeExcite(out_dim)

    def forward(self, x):  # x: (B, 3, 224, 224)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        x = x.mean(dim=(2, 3))  # GAP -> (B, out_dim)
        return x


class HybridViTCAB(nn.Module):
    """Full Stream 1 model: ViT branch + CAB branch -> fusion head -> logits."""

    def __init__(self, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        self.vit = ViTBranch()
        self.cab = CABBranch()
        self.fusion_fc = nn.Linear(EMBED_DIM * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_features: bool = False):
        vit_feat = self.vit(x)
        cab_feat = self.cab(x)
        fused = torch.cat([vit_feat, cab_feat], dim=1)  # (B, 128)
        h = torch.relu(self.fusion_fc(fused))
        h = self.dropout(h)
        logits = self.classifier(h)
        if return_features:
            return logits, h
        return logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = HybridViTCAB()
    n = count_params(m)
    print(f"HybridViTCAB params: {n:,} (paper reports 412,676)")
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    out = m(x)
    print("output shape:", out.shape)
