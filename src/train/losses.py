"""Class-weighted and focal loss, for pushing grass recall up specifically."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weighting.
    gamma=0 + alpha=None reduces to plain cross-entropy."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - target_probs) ** self.gamma
        loss = -focal_weight * target_log_probs
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


def class_weights_from_counts(counts: dict[str, int], class_order: list[str]) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1."""
    freqs = torch.tensor([counts[c] for c in class_order], dtype=torch.float32)
    weights = 1.0 / freqs
    weights = weights / weights.mean()
    return weights
