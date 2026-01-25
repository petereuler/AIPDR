import torch
import torch.nn as nn

from models.heading_classifier import FeatureExtractor, RegressorHead


class Navigator(nn.Module):
    """
    Navigator outputs [cos(phi), sin(phi), length, dz].
    """
    def __init__(self, imu_dim=6, feat_dim=64):
        super().__init__()
        self.extractor = FeatureExtractor(in_channels=imu_dim, feat_dim=feat_dim)
        self.output_head = RegressorHead(feat_dim, 4)

    def forward(self, imu_seq):
        feat = self.extractor(imu_seq)
        out = self.output_head(feat)
        return out
