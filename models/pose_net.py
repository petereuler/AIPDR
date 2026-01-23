import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rotation_matrix_from_6d(x, eps=1e-8):
    """
    Zhou et al. 6D rotation representation -> SO(3) via Gram-Schmidt.
    x: [B, 6]
    returns: [B, 3, 3]
    """
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]

    b1 = F.normalize(a1, dim=1, eps=eps)
    dot = torch.sum(b1 * a2, dim=1, keepdim=True)
    a2_ortho = a2 - dot * b1
    b2 = F.normalize(a2_ortho, dim=1, eps=eps)
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack([b1, b2, b3], dim=2)


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
    Estimate rotation matrix from IMU sequence using 6D rotation representation.
    """
    def __init__(self, imu_dim=6, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_size=imu_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 6)

    def forward(self, imu_seq):
        _, h = self.gru(imu_seq)
        h = h[-1]
        rot_6d = self.fc(h)
        R_pred = compute_rotation_matrix_from_6d(rot_6d)
        return R_pred
