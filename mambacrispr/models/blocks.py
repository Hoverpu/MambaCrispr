import torch
import torch.nn as nn


class MultiScaleTCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        return torch.cat([out3, out5, out7], dim=1)


class CSIM(nn.Module):
    """CNN-Sequential Interaction Module using cosine similarity."""

    def __init__(self, cnn_dim: int, rnn_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.cnn_to_q = nn.Linear(cnn_dim, hidden_dim)
        self.rnn_to_k = nn.Linear(rnn_dim, hidden_dim)
        self.rnn_to_v = nn.Linear(rnn_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, cnn_feat: torch.Tensor, rnn_feat: torch.Tensor) -> torch.Tensor:
        q = self.cnn_to_q(cnn_feat).unsqueeze(1)  # [B, 1, H]
        k = self.rnn_to_k(rnn_feat)                # [B, T, H]
        v = self.rnn_to_v(rnn_feat)                # [B, T, H]

        # Cosine similarity: [B, 1, T]
        cos_sim = nn.functional.cosine_similarity(q, k, dim=-1).unsqueeze(1)
        attn_weights = torch.softmax(cos_sim, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)  # [B, 1, H]
        return attended.squeeze(1)                 # [B, H]
