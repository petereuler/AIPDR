import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class SequenceRefiner(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_body_residual=0.05,
        max_rot_residual_deg=6.0,
    ):
        super().__init__()
        self.max_body_residual = float(max_body_residual)
        self.max_rot_residual_rad = float(max_rot_residual_deg) * math.pi / 180.0
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.body_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.rot_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        nn.init.zeros_(self.body_head[-1].weight)
        nn.init.zeros_(self.body_head[-1].bias)
        nn.init.zeros_(self.rot_head[-1].weight)
        nn.init.zeros_(self.rot_head[-1].bias)

    def forward(self, tokens):
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        x = self.input_proj(tokens)
        x = self.pos_encoder(x)
        seq_len = x.size(1)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        x = self.transformer(x, mask=mask)
        x = self.norm(x)
        body_residual = torch.tanh(self.body_head(x)) * self.max_body_residual
        rot_residual = torch.tanh(self.rot_head(x)) * self.max_rot_residual_rad
        if squeeze:
            return body_residual[0], rot_residual[0]
        return body_residual, rot_residual
