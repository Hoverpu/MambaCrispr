from typing import List

import numpy as np


ONEHOT_MAP = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

TOKEN_MAP = {"A": 2, "C": 3, "G": 4, "T": 5}


def dna_to_onehot(seq: str) -> np.ndarray:
    seq = seq.strip().upper()
    return np.array([ONEHOT_MAP.get(base, [0, 0, 0, 0]) for base in seq], dtype=np.float32)


def dna_to_tokens(seq: str, with_start_token: bool = True) -> List[int]:
    seq = seq.strip().upper()
    tokens = [TOKEN_MAP.get(base, 0) for base in seq]
    if with_start_token:
        return [1] + tokens
    return tokens
