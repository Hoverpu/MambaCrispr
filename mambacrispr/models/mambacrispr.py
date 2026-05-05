import torch
import torch.nn as nn
from mamba_ssm import Mamba

from .blocks import CSIM, MultiScaleTCN


class MambaCrispr(nn.Module):
    def __init__(self, interaction_type: str = "cross_attention"):
        super().__init__()
        self.interaction_type = interaction_type

        self.tcn1 = MultiScaleTCN(in_channels=4, out_channels=64, dropout=0.2)
        self.tcn2 = MultiScaleTCN(in_channels=192, out_channels=128, dropout=0.2)
        self.tcn_pool = nn.AdaptiveAvgPool1d(1)
        self.tcn_fc = nn.Sequential(
            nn.Linear(384, 320),
            nn.ELU(),
            nn.Dropout(0.2),
        )

        self.lstm_embedding = nn.Embedding(7, 60)
        self.branch_lstm = nn.LSTM(
            input_size=60,
            hidden_size=80,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.2,
        )
        self.lstm_fc = nn.Sequential(
            nn.Linear(3840, 320),
            nn.ELU(),
            nn.Dropout(0.2),
        )

        if interaction_type == "cross_attention":
            self.interaction = CSIM(cnn_dim=384, rnn_dim=160, hidden_dim=128)
            interaction_out_dim = 128
        else:
            self.interaction = None
            interaction_out_dim = 0

        fusion_in_dim = 320 + 320 + interaction_out_dim
        self.fusion_proj = nn.Linear(fusion_in_dim, 192)
        self.mamba = Mamba(d_model=192, d_state=12, d_conv=4, expand=2)
        self.final_fc = nn.Linear(192, 1)

    def forward(self, onehot_x: torch.Tensor, token_x: torch.Tensor) -> torch.Tensor:
        batch_size = onehot_x.size(0)
        if onehot_x.dim() == 4:
            onehot_x = onehot_x.squeeze(1).permute(0, 2, 1)
        elif onehot_x.dim() == 3:
            onehot_x = onehot_x.permute(0, 2, 1)

        tcn_out = self.tcn1(onehot_x)
        tcn_out = self.tcn2(tcn_out)
        tcn_global = self.tcn_pool(tcn_out).squeeze(-1)
        tcn_feat = self.tcn_fc(tcn_global)

        lstm_feat = self.lstm_embedding(token_x.long())
        lstm_seq, _ = self.branch_lstm(lstm_feat)
        lstm_out = lstm_seq.contiguous().view(batch_size, -1)
        lstm_out = self.lstm_fc(lstm_out)

        if self.interaction is not None:
            interaction_feat = self.interaction(tcn_global, lstm_seq)
            fused = torch.cat([tcn_feat, lstm_out, interaction_feat], dim=1)
        else:
            fused = torch.cat([tcn_feat, lstm_out], dim=1)

        mamba_in = self.fusion_proj(fused).unsqueeze(1)
        mamba_out = self.mamba(mamba_in).squeeze(1)
        return self.final_fc(mamba_out)
