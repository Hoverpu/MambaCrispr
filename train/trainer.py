import numpy as np
import torch
import torch.nn as nn


def train_epoch(model, loader, optimizer, criterion, device):
    """Standard training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for onehot_x, token_x, y in loader:
        onehot_x, token_x, y = onehot_x.to(device), token_x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(onehot_x, token_x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * onehot_x.size(0)

    return total_loss / len(loader.dataset)


def compute_dnlr_threshold(model, loader, device, noise_percentile):
    """Compute noise detection threshold from training set losses."""
    model.eval()
    losses = []

    with torch.no_grad():
        for onehot_x, token_x, y in loader:
            onehot_x, token_x, y = onehot_x.to(device), token_x.to(device), y.to(device)
            pred = model(onehot_x, token_x)
            losses.extend(torch.abs(pred - y).cpu().numpy().flatten())

    return np.percentile(np.array(losses), noise_percentile)


def dnlr_train_epoch(model, loader, optimizer, criterion, device, t_labels, alpha, threshold, in_warmup):
    """
    DNLR training epoch.

    Args:
        t_labels: mutable soft labels array (modified in-place for noisy samples)
        alpha: EMA coefficient for label update
        threshold: noise detection threshold (ignored during warmup)
        in_warmup: if True, use standard loss; if False, use DNLR

    Returns:
        (avg_loss, noisy_indices, predictions_dict)
    """
    model.train()
    total_loss = 0.0
    pred_cache = {}
    noisy_cache = {}

    for onehot_x, token_x, y, idx in loader:
        onehot_x, token_x, y = onehot_x.to(device), token_x.to(device), y.to(device)
        idx_np = idx.numpy()

        pred = model(onehot_x, token_x)
        pred_np = pred.detach().cpu().numpy().flatten()

        if in_warmup:
            loss = criterion(pred, y)
        else:
            # Use soft labels for training
            t_soft = torch.tensor(
                t_labels[idx_np], dtype=torch.float32
            ).unsqueeze(1).to(device)
            loss = criterion(pred, t_soft)

            # Detect noisy samples
            sample_loss = torch.abs(pred - y).detach().cpu().numpy().flatten()
            for i, gi in enumerate(idx_np):
                noisy_cache[gi] = sample_loss[i] > threshold

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * onehot_x.size(0)

        for i, gi in enumerate(idx_np):
            pred_cache[gi] = pred_np[i]

    return total_loss / len(loader.dataset), noisy_cache, pred_cache


def update_dnlr_labels(t_labels, noisy_cache, pred_cache, alpha):
    """EMA update of soft labels for noisy samples."""
    for gi, is_noisy in noisy_cache.items():
        if is_noisy:
            t_labels[gi] = alpha * t_labels[gi] + (1 - alpha) * pred_cache[gi]


class IndexedTensorDataset(torch.utils.data.TensorDataset):
    """TensorDataset that also returns the index."""
    def __getitem__(self, index):
        return super().__getitem__(index) + (index,)
