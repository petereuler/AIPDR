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


class SequenceConditionEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, seq):
        x = self.input_proj(seq)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x.mean(dim=1)


class TruthAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim=3,
        seq_len=64,
        latent_dim=64,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        condition_mode="none",
        imu_dim=6,
    ):
        super().__init__()
        condition_mode = str(condition_mode).lower()
        if condition_mode not in {"none", "encoder", "decoder", "both"}:
            raise ValueError(
                "condition_mode must be one of {'none', 'encoder', 'decoder', 'both'}, "
                f"got {condition_mode}"
            )
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.latent_dim = int(latent_dim)
        self.imu_dim = int(imu_dim)
        self.condition_mode = condition_mode
        self.use_imu_condition = condition_mode != "none"
        self.input_proj = nn.Linear(input_dim, d_model)
        self.encoder_pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, latent_dim)

        self.latent_to_tokens = nn.Linear(latent_dim, seq_len * d_model)
        self.decoder_pos = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, input_dim)

        if self.use_imu_condition:
            self.condition_encoder = SequenceConditionEncoder(
                input_dim=self.imu_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            if self.condition_mode in {"encoder", "both"}:
                self.encoder_condition_fuse = nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(d_model),
                )
            if self.condition_mode in {"decoder", "both"}:
                self.decoder_condition_fuse = nn.Sequential(
                    nn.Linear(latent_dim + d_model, latent_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(latent_dim),
                )

    def _resolve_condition(self, imu_seq=None, condition_feat=None):
        if not self.use_imu_condition:
            return None
        if condition_feat is not None:
            return condition_feat
        if imu_seq is None:
            raise ValueError("IMU condition is enabled but imu_seq was not provided")
        return self.condition_encoder(imu_seq)

    def encode(self, truth_seq, imu_seq=None, condition_feat=None):
        x = self.input_proj(truth_seq)
        x = self.encoder_pos(x)
        x = self.encoder(x)
        x = self.encoder_norm(x)
        pooled = x.mean(dim=1)
        if self.condition_mode in {"encoder", "both"}:
            cond = self._resolve_condition(imu_seq=imu_seq, condition_feat=condition_feat)
            pooled = self.encoder_condition_fuse(torch.cat([pooled, cond], dim=-1))
        return self.to_latent(pooled)

    def decode(self, latent, imu_seq=None, condition_feat=None):
        if self.condition_mode in {"decoder", "both"}:
            cond = self._resolve_condition(imu_seq=imu_seq, condition_feat=condition_feat)
            latent = self.decoder_condition_fuse(torch.cat([latent, cond], dim=-1))
        x = self.latent_to_tokens(latent).view(latent.size(0), self.seq_len, -1)
        x = self.decoder_pos(x)
        x = self.decoder(x)
        x = self.decoder_norm(x)
        return self.output_head(x)

    def forward(self, truth_seq, imu_seq=None):
        cond = self._resolve_condition(imu_seq=imu_seq) if self.use_imu_condition else None
        latent = self.encode(truth_seq, condition_feat=cond)
        recon = self.decode(latent, condition_feat=cond)
        return recon, latent
