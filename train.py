"""Variant 2 training: weighted cross-entropy with label smoothing, Adam + cosine, 20 epochs."""
import torch

from src.data.dataset import CLASSES
from src.train.losses import class_weights_from_counts
from variants.common import run_variant
from variants.v2_lightweight_streams.model import build_model


def build_loss(counts, device):
    w = class_weights_from_counts(dict(counts), CLASSES).to(device)
    ce = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    return lambda out, y: ce(out, y)


def build_optimizer(model, steps_per_epoch, max_epochs):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    return opt, sched, False


if __name__ == "__main__":
    run_variant(
        tag="v2_lightweight_streams",
        out_dir="variants/v2_lightweight_streams/outputs",
        build_model=lambda s: build_model(s, num_classes=len(CLASSES)),
        build_loss=build_loss,
        build_optimizer=build_optimizer,
        max_epochs=20,
    )
