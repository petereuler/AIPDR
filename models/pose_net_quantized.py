import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=1)


def quat_norm(q):
    return q / (q.norm(dim=1, keepdim=True) + 1e-8)


def quat_to_rotmat(q):
    q = quat_norm(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz

    return torch.stack(
        [torch.stack([r00, r01, r02], dim=1),
         torch.stack([r10, r11, r12], dim=1),
         torch.stack([r20, r21, r22], dim=1)],
        dim=1
    )


def euler_to_quat_xyz(roll, pitch, yaw):
    """
    roll, pitch, yaw: [B, 1] radians
    returns: [B, 4] quaternion (w, x, y, z)
    """
    hr = roll * 0.5
    hp = pitch * 0.5
    hy = yaw * 0.5
    cr, sr = torch.cos(hr), torch.sin(hr)
    cp, sp = torch.cos(hp), torch.sin(hp)
    cy, sy = torch.cos(hy), torch.sin(hy)

    # q = qz * qy * qx
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = cy * sp * cr + sy * cp * sr
    qz = sy * cp * cr - cy * sp * sr
    q = torch.cat([qw, qx, qy, qz], dim=1)
    return quat_norm(q)


def quat_to_euler_xyz(q):
    """
    q: [B, 4] quaternion (w, x, y, z)
    returns: roll, pitch, yaw in radians, each [B, 1]
    """
    q = quat_norm(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return roll.unsqueeze(1), pitch.unsqueeze(1), yaw.unsqueeze(1)


def quat_to_quantized_labels(q_rel, bins):
    """
    Convert relative quaternion to quantized labels per axis.
    q_rel: [B, 4]
    bins: [K] tensor (radians)
    returns: labels [B, 3] (roll, pitch, yaw indices)
    """
    roll, pitch, yaw = quat_to_euler_xyz(q_rel)
    angles = torch.cat([roll, pitch, yaw], dim=1)  # [B, 3]
    # find nearest bin index
    diff = angles.unsqueeze(-1) - bins.view(1, 1, -1)
    labels = torch.argmin(torch.abs(diff), dim=-1)
    return labels


class QuantizedPoseHead(nn.Module):
    """
    Three-axis classification head with soft expectation decoding.
    """
    def __init__(self, feat_dim, num_bins=256, angle_range_deg=180.0):
        super().__init__()
        self.num_bins = num_bins
        self.angle_range_deg = angle_range_deg
        self.fc_roll = nn.Linear(feat_dim, num_bins)
        self.fc_pitch = nn.Linear(feat_dim, num_bins)
        self.fc_yaw = nn.Linear(feat_dim, num_bins)

        # Bin centers in radians
        angle_range_rad = math.radians(angle_range_deg)
        bins = torch.linspace(-angle_range_rad, angle_range_rad, num_bins)
        self.register_buffer("bins", bins)

    def forward(self, feat):
        logits_roll = self.fc_roll(feat)
        logits_pitch = self.fc_pitch(feat)
        logits_yaw = self.fc_yaw(feat)

        # Soft expectation: sum(softmax * bin_values)
        # This keeps gradients and lets the model "lock to zero" with peaked distributions.
        prob_roll = F.softmax(logits_roll, dim=1)
        prob_pitch = F.softmax(logits_pitch, dim=1)
        prob_yaw = F.softmax(logits_yaw, dim=1)

        roll = torch.sum(prob_roll * self.bins.unsqueeze(0), dim=1, keepdim=True)
        pitch = torch.sum(prob_pitch * self.bins.unsqueeze(0), dim=1, keepdim=True)
        yaw = torch.sum(prob_yaw * self.bins.unsqueeze(0), dim=1, keepdim=True)

        # Convert to quaternion delta
        q_pred = euler_to_quat_xyz(roll, pitch, yaw)
        return {
            "logits": (logits_roll, logits_pitch, logits_yaw),
            "angles": (roll, pitch, yaw),
            "q_pred": q_pred,
        }


class QuantizedPoseNet(nn.Module):
    """
    Transformer backbone + quantized pose head.
    Forward returns dict with logits and q_pred.
    """
    def __init__(self, imu_dim=6, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1,
                 num_bins=256, angle_range_deg=180.0):
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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = QuantizedPoseHead(feat_dim=d_model, num_bins=num_bins, angle_range_deg=angle_range_deg)

    def forward(self, imu_seq):
        x = self.input_proj(imu_seq)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        feat = x.mean(dim=1)
        return self.head(feat)

    def forward_rotmat(self, imu_seq):
        out = self.forward(imu_seq)
        return quat_to_rotmat(out["q_pred"])


class PhysicsAwareLoss(nn.Module):
    """
    Combined classification + geodesic loss.
    L_total = lambda_cls * L_CE + lambda_geo * L_geo
    """
    def __init__(self, lambda_cls=1.0, lambda_geo=1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_geo = lambda_geo

    def forward(self, logits_tuple, q_pred, q_gt, label_tuple):
        logits_roll, logits_pitch, logits_yaw = logits_tuple
        label_roll, label_pitch, label_yaw = label_tuple

        loss_cls = (
            F.cross_entropy(logits_roll, label_roll) +
            F.cross_entropy(logits_pitch, label_pitch) +
            F.cross_entropy(logits_yaw, label_yaw)
        )

        q_pred = quat_norm(q_pred)
        q_gt = quat_norm(q_gt)
        dot = torch.abs(torch.sum(q_pred * q_gt, dim=1))
        loss_geo = torch.mean(1.0 - dot)

        return self.lambda_cls * loss_cls + self.lambda_geo * loss_geo
