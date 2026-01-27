import torch
import torch.nn as nn

from models.heading_classifier import FeatureExtractor, RegressorHead


class Navigator(nn.Module):
    """
    Navigator directly regresses delta position (dx, dy, dz) in aligned world frame.
    """
    def __init__(self, imu_dim=6, feat_dim=64):
        super().__init__()
        self.extractor = FeatureExtractor(in_channels=imu_dim, feat_dim=feat_dim)
        self.output_head = RegressorHead(feat_dim, 3)

    def forward(self, imu_seq):
        feat = self.extractor(imu_seq)
        delta_p = self.output_head(feat)
        return delta_p
