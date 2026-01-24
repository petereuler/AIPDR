import torch
import torch.nn as nn

from models.heading_classifier import FeatureExtractor, RegressorHead


class Navigator(nn.Module):
    """
    Navigator with a single delta-position head in world coordinates.
    Forward returns delta_p_xyz.
    """
    def __init__(self, imu_dim=6, feat_dim=64):
        super().__init__()
        self.extractor = FeatureExtractor(in_channels=imu_dim, feat_dim=feat_dim)
        self.dp_head = RegressorHead(feat_dim, 3)

    def forward(self, imu_seq):
        feat = self.extractor(imu_seq)
        delta_p = self.dp_head(feat)
        return delta_p
