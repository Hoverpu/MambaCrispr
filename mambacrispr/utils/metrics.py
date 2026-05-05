from typing import Tuple

import numpy as np
import scipy.stats as stats
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader


def evaluate_regression(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for onehot_x, token_x, y in loader:
            onehot_x = onehot_x.to(device)
            token_x = token_x.to(device)
            y = y.to(device)
            pred = model(onehot_x, token_x)
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true).reshape(-1)
    y_pred = np.concatenate(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    spearman = stats.spearmanr(y_true, y_pred)[0]
    pearson = stats.pearsonr(y_true, y_pred)[0]
    r2 = r2_score(y_true, y_pred)
    return float(mse), float(spearman), float(pearson), float(r2)
