"""Fusion heads combining the 4 stream feature vectors (each 64-d) into a classification.

Two variants, per the task spec's ablation:
  - ConcatFusion: naive concatenation -> Dense -> Dropout -> Dense(num_classes)
  - AttentionFusion: learned per-sample per-stream gating weights (softmax over streams),
    weighted sum, then -> Dense -> Dropout -> Dense(num_classes)

MultiStreamModel wires up however many of the 4 streams are enabled (so we can run the
incremental ablation: RGB-only -> +veg -> +texture -> +CLIP) with either fusion head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

STREAM_DIM = 64


class ConcatFusion(nn.Module):
    def __init__(self, num_streams: int, num_classes: int = 4, dropout: float = 0.4, hidden: int = 128):
        super().__init__()
        in_dim = num_streams * STREAM_DIM
        self.fc = nn.Linear(in_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, feats: list[torch.Tensor]):
        x = torch.cat(feats, dim=1)
        h = torch.relu(self.fc(x))
        h = self.dropout(h)
        return self.classifier(h)


class AttentionFusion(nn.Module):
    """Per-sample, per-stream gating: a small MLP over the concatenated features produces
    softmax weights over the streams, which rescale each stream's feature vector before
    concatenation. Lets the model down-weight a stream that's uninformative for a given sample."""

    def __init__(self, num_streams: int, num_classes: int = 4, dropout: float = 0.4, hidden: int = 128):
        super().__init__()
        self.num_streams = num_streams
        gate_in = num_streams * STREAM_DIM
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_streams),
        )
        self.fc = nn.Linear(gate_in, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, feats: list[torch.Tensor]):
        concat = torch.cat(feats, dim=1)
        weights = torch.softmax(self.gate(concat), dim=1)  # (B, num_streams)
        weighted = [feats[i] * weights[:, i : i + 1] for i in range(self.num_streams)]
        x = torch.cat(weighted, dim=1)
        h = torch.relu(self.fc(x))
        h = self.dropout(h)
        return self.classifier(h)


class MultiStreamModel(nn.Module):
    """streams: dict of name -> nn.Module branch, each mapping its raw input to a (B, 64) vector.
    forward() takes a dict of name -> input tensor and returns logits (+ optional attn weights)."""

    def __init__(self, streams: dict[str, nn.Module], num_classes: int = 4, fusion: str = "concat"):
        super().__init__()
        self.stream_names = list(streams.keys())
        self.branches = nn.ModuleDict(streams)
        n = len(streams)
        if fusion == "concat":
            self.fusion = ConcatFusion(n, num_classes)
        elif fusion == "attention":
            self.fusion = AttentionFusion(n, num_classes)
        else:
            raise ValueError(f"unknown fusion type: {fusion}")

    def forward(self, inputs: dict[str, torch.Tensor]):
        feats = [self.branches[name](inputs[name]) for name in self.stream_names]
        return self.fusion(feats)
