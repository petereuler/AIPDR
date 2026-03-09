import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class ResidualBlockDown1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, feat_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock1D(hidden_dim, dropout=dropout),
            ResidualBlock1D(hidden_dim, dropout=dropout),
        )
        self.layer2 = nn.Sequential(
            ResidualBlockDown1D(hidden_dim, hidden_dim * 2, stride=2, dropout=dropout),
            ResidualBlock1D(hidden_dim * 2, dropout=dropout),
        )
        self.layer3 = nn.Sequential(
            ResidualBlockDown1D(hidden_dim * 2, hidden_dim * 4, stride=2, dropout=dropout),
            ResidualBlock1D(hidden_dim * 4, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class Navigator(nn.Module):
    def __init__(self, imu_dim=6, feat_dim=64):
        super().__init__()
        self.extractor = FeatureExtractor(in_channels=imu_dim, feat_dim=feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, imu_seq):
        return self.head(self.extractor(imu_seq))
