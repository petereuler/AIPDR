import torch
import torch.nn as nn

from models.posenet import PositionalEncoding


class IMUToPoseLatent(nn.Module):
    def __init__(
        self,
        imu_dim=6,
        latent_dim=64,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(imu_dim, d_model)
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
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, latent_dim),
        )

    def encode_features(self, imu_seq):
        x = self.input_proj(imu_seq)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, imu_seq):
        return self.head(self.encode_features(imu_seq))
