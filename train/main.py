import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adamax
from torch.utils.data import DataLoader, TensorDataset

from train.config import (
    DATASETS, DATA_DIR, SEED, LR, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE,
    MAX_EPOCHS, EARLY_STOP_PATIENCE,
    DNLR_ALPHA, DNLR_WARMUP_PATIENCE, DNLR_NOISE_PERCENTILE,
)
from train.data import load_data, build_dataloaders
from train.evaluate import evaluate_model
from train.trainer import (
    train_epoch, dnlr_train_epoch, compute_dnlr_threshold,
    update_dnlr_labels, IndexedTensorDataset,
)
from mambacrispr.models import MambaCrispr
from mambacrispr.utils.seed import set_seed


def train_standard(model, train_loader, val_loader, test_loader, device, dataset_name):
    """Standard training with early stopping on val spearman."""
    optimizer = Adamax(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_sp = -1
    best_metrics = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, val_sp, val_pe, val_r2 = evaluate_model(model, val_loader, device)
        test_mse, test_sp, test_pe, test_r2 = evaluate_model(model, test_loader, device)

        print(f"[{dataset_name}] Epoch {epoch:03d} | Loss={avg_loss:.4f} | "
              f"Val SP={val_sp:.4f} | Test SP={test_sp:.4f}")

        if val_sp > best_val_sp:
            best_val_sp = val_sp
            best_metrics = (test_mse, test_sp, test_pe, test_r2)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    return best_metrics


def train_dnlr(model, train_loader, val_loader, test_loader, device, dataset_name, train_labels):
    """DNLR training: warmup → DNLR phase."""
    optimizer = Adamax(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    t_labels = train_labels.copy()
    in_warmup = True
    warmup_best_sp = -1
    warmup_no_improve = 0
    warmup_end_epoch = None
    best_metrics = None

    for epoch in range(1, MAX_EPOCHS + 1):

        # Compute DNLR threshold (only after warmup)
        threshold = None
        if not in_warmup:
            threshold = compute_dnlr_threshold(model, train_loader, device, DNLR_NOISE_PERCENTILE)

        # Train epoch
        avg_loss, noisy_cache, pred_cache = dnlr_train_epoch(
            model, train_loader, optimizer, criterion, device,
            t_labels, DNLR_ALPHA, threshold, in_warmup
        )

        # EMA label update
        if not in_warmup:
            update_dnlr_labels(t_labels, noisy_cache, pred_cache, DNLR_ALPHA)

        # Evaluate
        val_mse, val_sp, val_pe, val_r2 = evaluate_model(model, val_loader, device)
        test_mse, test_sp, test_pe, test_r2 = evaluate_model(model, test_loader, device)

        stage = "Warmup" if in_warmup else "DNLR"
        print(f"[{dataset_name}] Epoch {epoch:03d} [{stage}] | Loss={avg_loss:.4f} | "
              f"Val SP={val_sp:.4f} | Test SP={test_sp:.4f}")

        # Track best metrics
        if best_metrics is None or val_sp > best_metrics[1]:
            best_metrics = (test_mse, test_sp, test_pe, test_r2)

        # Warmup control
        if in_warmup:
            if val_sp > warmup_best_sp:
                warmup_best_sp = val_sp
                warmup_no_improve = 0
            else:
                warmup_no_improve += 1

            if warmup_no_improve >= DNLR_WARMUP_PATIENCE:
                in_warmup = False
                warmup_end_epoch = epoch
                print(f"  Warmup → DNLR at epoch {epoch}")

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="MambaCrispr Training")
    parser.add_argument("--dnlr", action="store_true", default=False,
                        help="Enable DNLR training framework")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {'DNLR' if args.dnlr else 'Standard'}")

    for dataset_name in DATASETS:
        csv_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
        if not os.path.exists(csv_path):
            print(f"  Skipping {dataset_name}: file not found")
            continue

        print(f"\nDataset: {dataset_name}")

        # Load and split data
        onehot, tokens, labels = load_data(csv_path)
        train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = \
            build_dataloaders(onehot, tokens, labels, seed=SEED)

        print(f"  Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

        # Create model
        model = MambaCrispr().to(device)

        # Train
        if args.dnlr:
            train_dataset = IndexedTensorDataset(
                torch.tensor(onehot[train_idx], dtype=torch.float32),
                torch.tensor(tokens[train_idx], dtype=torch.long),
                torch.tensor(labels[train_idx], dtype=torch.float32).unsqueeze(1),
            )
            dnlr_train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            best_metrics = train_dnlr(
                model, dnlr_train_loader, val_loader, test_loader,
                device, dataset_name, labels[train_idx]
            )
        else:
            best_metrics = train_standard(
                model, train_loader, val_loader, test_loader,
                device, dataset_name
            )

        test_mse, test_sp, test_pe, test_r2 = best_metrics
        print(f"  Best: {dataset_name} | Spearman={test_sp:.4f} | Pearson={test_pe:.4f} | MSE={test_mse:.4f} | R2={test_r2:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
