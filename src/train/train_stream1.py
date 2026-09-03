"""Train Stream 1 (paper's ViT+CAB reproduction) standalone, to validate the pipeline
against the paper's reported ~88% accuracy / 0.97 AUC before adding more streams.

Paper's training recipe: Adam, sparse categorical crossentropy, cosine annealing LR
(1e-3 -> 1e-5), batch 32, max 25 epochs, early stopping patience 5 on val loss.
The paper doesn't specify a separate val split, so we carve 10% off the train set
(stratified, seed 42) for early stopping and keep the held-out test set untouched.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.data.dataset import CLASSES, WeedDataset, read_manifest
from src.eval.metrics import compute_metrics, format_report
from src.models.streams.vit_cab import HybridViTCAB, count_params

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 25
PATIENCE = 5
LR_MAX = 1e-3
LR_MIN = 1e-5
VAL_FRAC = 0.10
CKPT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results/tables")

torch.manual_seed(SEED)
np.random.seed(SEED)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # -> [0,1], CHW
    ]
)
EVAL_TRANSFORM = transforms.Compose([transforms.ToTensor()])


def make_train_val_split(seed: int = SEED):
    items = read_manifest("train")
    labels = [c for _, c in items]
    idx = list(range(len(items)))
    train_idx, val_idx = train_test_split(idx, test_size=VAL_FRAC, random_state=seed, stratify=labels)
    return train_idx, val_idx


def run_epoch(model, loader, device, optimizer=None, criterion=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0
    all_true, all_pred, all_proba = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
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
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_proba = np.concatenate(all_proba)
    metrics = compute_metrics(all_true, all_pred, all_proba)
    metrics["loss"] = total_loss / n
    return metrics


def main():
    device = get_device()
    print(f"Using device: {device}")

    full_train = WeedDataset("train", transform=None)
    train_idx, val_idx = make_train_val_split()
    train_ds = Subset(WeedDataset("train", transform=TRAIN_TRANSFORM), train_idx)
    val_ds = Subset(WeedDataset("train", transform=EVAL_TRANSFORM), val_idx)
    test_ds = WeedDataset("test", transform=EVAL_TRANSFORM)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = HybridViTCAB(num_classes=len(CLASSES)).to(device)
    print(f"Stream1 params: {count_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CKPT_DIR / "stream1_vit_cab_best.pt"

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, criterion)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None, criterion=criterion)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"epoch {epoch:2d}/{MAX_EPOCHS}  lr={lr_now:.2e}  "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']*100:.2f}%  "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']*100:.2f}%"
        )
        history.append(
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
            }
        )

        if val_metrics["loss"] < best_val_loss - 1e-5:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val_loss improvement for {PATIENCE} epochs).")
                break

    elapsed = time.time() - t0
    print(f"Training done in {elapsed/60:.1f} min. Best val_loss={best_val_loss:.4f}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = run_epoch(model, test_loader, device, optimizer=None, criterion=criterion)
    report = format_report(test_metrics, title="Stream 1 (ViT+CAB reproduction) - TEST SET")
    print("\n" + report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "stream1_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(RESULTS_DIR / "stream1_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(RESULTS_DIR / "stream1_report.txt", "w") as f:
        f.write(report + "\n")

    print(f"\nParams: {count_params(model):,} (paper: 412,676)")
    print("Saved: checkpoints/stream1_vit_cab_best.pt, results/tables/stream1_*")


if __name__ == "__main__":
    main()
