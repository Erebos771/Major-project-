"""Fine-tune a pretrained ConvNeXt-Tiny (ImageNet weights) as an alternative to / comparison
point against the from-scratch Stream 1 ViT+CAB, since compute is not a constraint here.

Usage:
    python -m src.train.train_pretrained_backbone --tag convnext_tiny_finetune
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from src.data.dataset import CLASSES, WeedDataset, read_manifest
from src.eval.metrics import compute_metrics, format_report

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 25
PATIENCE = 5
LR_MAX = 3e-4  # lower LR for fine-tuning a pretrained backbone
LR_MIN = 1e-6
VAL_FRAC = 0.10

torch.manual_seed(SEED)
np.random.seed(SEED)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes: int = 4) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, loader, device, optimizer=None, criterion=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0
    all_true, all_pred, all_proba = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(imgs)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_proba.append(proba)
        all_pred.append(proba.argmax(axis=1))
        all_true.append(labels.cpu().numpy())
    all_true, all_pred, all_proba = np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_proba)
    metrics = compute_metrics(all_true, all_pred, all_proba)
    metrics["loss"] = total_loss / n
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="convnext_tiny_finetune")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )

    items = read_manifest("train")
    labels_str = [c for _, c in items]
    idx = list(range(len(items)))
    train_idx, val_idx = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=labels_str)

    train_ds = Subset(WeedDataset("train", transform=train_tf), train_idx)
    val_ds = Subset(WeedDataset("train", transform=eval_tf), val_idx)
    test_ds = WeedDataset("test", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(len(CLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ConvNeXt-Tiny params: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)

    ckpt_path = Path("checkpoints") / f"{args.tag}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss, no_improve = float("inf"), 0
    history = []

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_m = run_epoch(model, train_loader, device, optimizer, criterion)
        val_m = run_epoch(model, val_loader, device, optimizer=None, criterion=criterion)
        scheduler.step()
        print(
            f"[{args.tag}] epoch {epoch:2d}/{MAX_EPOCHS}  lr={scheduler.get_last_lr()[0]:.2e}  "
            f"train_acc={train_m['accuracy']*100:.2f}%  val_acc={val_m['accuracy']*100:.2f}%  "
            f"val_loss={val_m['loss']:.4f}"
        )
        history.append({"epoch": epoch, "train_acc": train_m["accuracy"], "val_acc": val_m["accuracy"], "val_loss": val_m["loss"]})
        if val_m["loss"] < best_val_loss - 1e-5:
            best_val_loss, no_improve = val_m["loss"], 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[{args.tag}] Early stopping at epoch {epoch}.")
                break

    print(f"[{args.tag}] Training done in {(time.time()-t0)/60:.1f} min.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = run_epoch(model, test_loader, device, optimizer=None, criterion=criterion)
    report = format_report(test_metrics, title=f"{args.tag} - TEST SET")
    print("\n" + report)

    results_dir = Path("results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{args.tag}_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(results_dir / f"{args.tag}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(results_dir / f"{args.tag}_report.txt", "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
