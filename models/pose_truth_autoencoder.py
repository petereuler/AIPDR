import torch
import torch.nn as nn
import torch.nn.functional as F

from models.truth_autoencoder import TruthAutoEncoder


def quat_normalize_torch(q):
    return q / (q.norm(dim=-1, keepdim=True) + 1e-8)


class PoseTruthAutoEncoder(TruthAutoEncoder):
    def __init__(
        self,
        seq_len=64,
        latent_dim=64,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__(
            input_dim=4,
            seq_len=seq_len,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def decode(self, latent):
        return quat_normalize_torch(super().decode(latent))

    def forward(self, truth_seq):
        truth_seq = quat_normalize_torch(truth_seq)
        latent = self.encode(truth_seq)
        recon = self.decode(latent)
        return recon, latent


def quaternion_sequence_loss(q_pred, q_gt):
    q_pred = quat_normalize_torch(q_pred)
    q_gt = quat_normalize_torch(q_gt)
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=-1))
    return torch.mean(1.0 - dot)


def quaternion_endpoint_error_rad(q_pred, q_gt):
    q_pred = quat_normalize_torch(q_pred[:, -1, :])
    q_gt = quat_normalize_torch(q_gt[:, -1, :])
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=-1)).clamp(-1.0, 1.0)
    return torch.mean(2.0 * torch.arccos(dot))
