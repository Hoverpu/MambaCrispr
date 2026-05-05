import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from train.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE
from mambacrispr.utils.encoding import dna_to_onehot, dna_to_tokens


def load_data(csv_path: str, seq_len: int = 23):
    """Load CSV and return (onehot, tokens, labels)."""
    df = pd.read_csv(csv_path)

    # Handle both column name formats
    if "sgRNA" in df.columns and "indel" in df.columns:
        seq_col, label_col = "sgRNA", "indel"
    elif "DNA_Sequence" in df.columns and "Indel" in df.columns:
        seq_col, label_col = "DNA_Sequence", "Indel"
    else:
        raise ValueError(f"CSV must have (sgRNA, indel) or (DNA_Sequence, Indel) columns: {csv_path}")

    seqs = [str(s).strip().upper()[:seq_len] for s in df[seq_col].tolist()]
    labels = df[label_col].astype(np.float32).to_numpy()

    onehot = np.array([dna_to_onehot(s) for s in seqs], dtype=np.float32)
    onehot = np.reshape(onehot, (-1, 1, seq_len, 4))

    tokens = np.array([dna_to_tokens(s, with_start_token=True) for s in seqs], dtype=np.int64)

    return onehot, tokens, labels


def build_dataloaders(
    onehot: np.ndarray,
    tokens: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
):
    """Split data into train/val/test (76.5/8.5/15) and return DataLoaders."""
    N = len(labels)
    idx = np.arange(N)

    # Stage 1: 85% train+val, 15% test
    trainval_idx, test_idx = train_test_split(
        idx, test_size=TEST_RATIO, random_state=seed, shuffle=True
    )
    # Stage 2: 90% of train+val → train, 10% → val
    # 0.9 * 0.85 = 0.765, 0.1 * 0.85 = 0.085
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO), random_state=seed, shuffle=True
    )

    def make_dataset(ind):
        return TensorDataset(
            torch.tensor(onehot[ind], dtype=torch.float32),
            torch.tensor(tokens[ind], dtype=torch.long),
            torch.tensor(labels[ind], dtype=torch.float32).unsqueeze(1),
        )

    train_loader = DataLoader(make_dataset(train_idx), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(make_dataset(val_idx), batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(make_dataset(test_idx), batch_size=TEST_BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx
