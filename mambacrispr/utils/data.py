from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .encoding import dna_to_onehot, dna_to_tokens


def _resolve_columns(df: pd.DataFrame) -> Tuple[str, str]:
    if "sgRNA" in df.columns and "indel" in df.columns:
        return "sgRNA", "indel"
    if "DNA_Sequence" in df.columns and "Indel" in df.columns:
        return "DNA_Sequence", "Indel"
    raise ValueError("CSV must include either (sgRNA, indel) or (DNA_Sequence, Indel) columns.")


def load_hnn_csv(csv_path: str, seq_len: int = 23) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    seq_col, label_col = _resolve_columns(df)

    seqs = [str(s).strip().upper()[:seq_len] for s in df[seq_col].tolist()]
    labels = df[label_col].astype(np.float32).to_numpy()

    onehot = np.array([dna_to_onehot(s) for s in seqs], dtype=np.float32)
    onehot = np.reshape(onehot, (-1, 1, seq_len, 4))

    tokens = np.array([dna_to_tokens(s, with_start_token=True) for s in seqs], dtype=np.int64)
    return onehot, tokens, labels


def build_dataloader_triplet(
    onehot: np.ndarray,
    tokens: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    train_batch_size: int = 16,
    test_batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=random_state, shuffle=True
    )

    train_set = TensorDataset(
        torch.tensor(onehot[train_idx], dtype=torch.float32),
        torch.tensor(tokens[train_idx], dtype=torch.long),
        torch.tensor(labels[train_idx], dtype=torch.float32).unsqueeze(1),
    )
    test_set = TensorDataset(
        torch.tensor(onehot[test_idx], dtype=torch.float32),
        torch.tensor(tokens[test_idx], dtype=torch.long),
        torch.tensor(labels[test_idx], dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader, train_idx, test_idx
