import torch
import torch.nn as nn

from models.heading_classifier import FeatureExtractor, RegressorHead


class Navigator(nn.Module):
    """
    Navigator with three heads: step length, heading angle, vertical delta.
    Forward returns (len, heading, dz, delta_p_xyz).
    """
    def __init__(self, imu_dim=6, feat_dim=64):
        super().__init__()
        self.extractor = FeatureExtractor(in_channels=imu_dim, feat_dim=feat_dim)
        self.len_head = RegressorHead(feat_dim, 1)
        self.heading_head = RegressorHead(feat_dim, 1)
        self.vert_head = RegressorHead(feat_dim, 1)
        self.loss_w_len = nn.Parameter(torch.tensor(1.0))
        self.loss_w_head = nn.Parameter(torch.tensor(1.0))
        self.loss_w_dp = nn.Parameter(torch.tensor(1.0))

    def forward(self, imu_seq):
        feat = self.extractor(imu_seq)
        step_len = self.len_head(feat)
        heading = self.heading_head(feat)
        dz = self.vert_head(feat)

        dir_xy = torch.cat([torch.cos(heading), torch.sin(heading)], dim=1)
        dxy = step_len * dir_xy
        delta_p = torch.cat([dxy, dz], dim=1)
        return step_len, heading, dz, delta_p

    def normalized_loss_weights(self):
        weights = torch.stack([self.loss_w_len, self.loss_w_head, self.loss_w_dp])
        weights = torch.nn.functional.softplus(weights)
        return weights / (weights.sum() + 1e-8)
