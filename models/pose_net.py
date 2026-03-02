import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def quat_to_rotmat(q):
    """
    Quaternion (w, x, y, z) -> rotation matrix [B, 3, 3]
    """
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
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


def quat_conj(q):
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=1)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=1)


def rotmat_to_quat(R):
    """
    Rotation matrix [B, 3, 3] -> quaternion (w, x, y, z).
    """
    r00 = R[:, 0, 0]
    r01 = R[:, 0, 1]
    r02 = R[:, 0, 2]
    r10 = R[:, 1, 0]
    r11 = R[:, 1, 1]
    r12 = R[:, 1, 2]
    r20 = R[:, 2, 0]
    r21 = R[:, 2, 1]
    r22 = R[:, 2, 2]

    q = torch.zeros((R.size(0), 4), device=R.device, dtype=R.dtype)
    t = r00 + r11 + r22

    mask = t > 0.0
    if mask.any():
        s = torch.sqrt(t[mask] + 1.0) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (r21[mask] - r12[mask]) / s
        q[mask, 2] = (r02[mask] - r20[mask]) / s
        q[mask, 3] = (r10[mask] - r01[mask]) / s

    mask1 = (~mask) & (r00 > r11) & (r00 > r22)
    if mask1.any():
        s = torch.sqrt(1.0 + r00[mask1] - r11[mask1] - r22[mask1]) * 2.0
        q[mask1, 0] = (r21[mask1] - r12[mask1]) / s
        q[mask1, 1] = 0.25 * s
        q[mask1, 2] = (r01[mask1] + r10[mask1]) / s
        q[mask1, 3] = (r02[mask1] + r20[mask1]) / s

    mask2 = (~mask) & (~mask1) & (r11 > r22)
    if mask2.any():
        s = torch.sqrt(1.0 + r11[mask2] - r00[mask2] - r22[mask2]) * 2.0
        q[mask2, 0] = (r02[mask2] - r20[mask2]) / s
        q[mask2, 1] = (r01[mask2] + r10[mask2]) / s
        q[mask2, 2] = 0.25 * s
        q[mask2, 3] = (r12[mask2] + r21[mask2]) / s

    mask3 = (~mask) & (~mask1) & (~mask2)
    if mask3.any():
        s = torch.sqrt(1.0 + r22[mask3] - r00[mask3] - r11[mask3]) * 2.0
        q[mask3, 0] = (r10[mask3] - r01[mask3]) / s
        q[mask3, 1] = (r02[mask3] + r20[mask3]) / s
        q[mask3, 2] = (r12[mask3] + r21[mask3]) / s
        q[mask3, 3] = 0.25 * s

    return q / (q.norm(dim=1, keepdim=True) + 1e-8)


def euler_to_quat_xyz(roll, pitch, yaw):
    hr = roll * 0.5
    hp = pitch * 0.5
    hy = yaw * 0.5
    cr, sr = torch.cos(hr), torch.sin(hr)
    cp, sp = torch.cos(hp), torch.sin(hp)
    cy, sy = torch.cos(hy), torch.sin(hy)
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = cy * sp * cr + sy * cp * sr
    qz = sy * cp * cr - cy * sp * sr
    q = torch.cat([qw, qx, qy, qz], dim=1)
    return q / (q.norm(dim=1, keepdim=True) + 1e-8)


def quat_to_euler_xyz(q):
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
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
    return torch.stack([roll, pitch, yaw], dim=1)


def rotate_imu(imu_seq, R_pred):
    """
    imu_seq: [B, T, 6] (gyro, acc) in body frame
    R_pred: [B, 3, 3] (body -> world)
    returns: imu_global [B, T, 6]
    """
    gyro_local = imu_seq[:, :, 0:3]
    acc_local = imu_seq[:, :, 3:6]
    R_t = R_pred.transpose(1, 2)
    gyro_global = torch.matmul(gyro_local, R_t)
    acc_global = torch.matmul(acc_local, R_t)
    return torch.cat([gyro_global, acc_global], dim=2)


class PoseNet(nn.Module):
    """
    Estimate rotation matrix from IMU sequence using quaternion representation.
    """
    def __init__(self, imu_dim=6, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_size=imu_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, imu_seq):
        _, h = self.gru(imu_seq)
        h = h[-1]
        q_pred = F.normalize(self.fc(h), dim=1, eps=1e-8)
        return q_pred


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


class PoseNetTransformer(nn.Module):
    """
    Transformer-based pose estimator. Input: [B, T, 6], Output: [B, 4] quaternion.
    """
    def __init__(self, imu_dim=6, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
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
        self.fc = nn.Linear(d_model, 4)

    def forward(self, imu_seq):
        x = self.input_proj(imu_seq)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x_mean = x.mean(dim=1)
        out = self.fc(x_mean)
        q_pred = F.normalize(out[:, 0:4], dim=1, eps=1e-8)
        return q_pred
