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
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.latent_dim = int(latent_dim)
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

    def encode(self, truth_seq):
        x = self.input_proj(truth_seq)
        x = self.encoder_pos(x)
        x = self.encoder(x)
        x = self.encoder_norm(x)
        pooled = x.mean(dim=1)
        return self.to_latent(pooled)

    def decode(self, latent):
        x = self.latent_to_tokens(latent).view(latent.size(0), self.seq_len, -1)
        x = self.decoder_pos(x)
        x = self.decoder(x)
        x = self.decoder_norm(x)
        return self.output_head(x)

    def forward(self, truth_seq):
        latent = self.encode(truth_seq)
        recon = self.decode(latent)
        return recon, latent
