import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pose_net import quat_to_rotmat


class PhysicsAwareHeadingNet(nn.Module):
    """
    End-to-end physics-aware heading predictor with a pose stream and heading stream.
    Input: imu_seq [B, T, 6] (gyro[x,y,z], acc[x,y,z]) in body frame.
    Output: heading_pred, R_pred, acc_global
    """
    def __init__(self, imu_dim=6, pose_hidden=128, heading_hidden=128, heading_out_dim=2):
        super().__init__()
        self.imu_dim = imu_dim
        self.pose_gru = nn.GRU(input_size=imu_dim, hidden_size=pose_hidden, batch_first=True)
        self.pose_fc = nn.Linear(pose_hidden, 4)

        self.heading_gru = nn.GRU(input_size=imu_dim, hidden_size=heading_hidden, batch_first=True)
        self.heading_fc = nn.Linear(heading_hidden, heading_out_dim)

    def forward(self, imu_seq):
        """
        imu_seq: [B, T, 6] where last dim is [gyro, acc]
        Returns:
            heading_pred: [B, heading_out_dim]
            R_pred: [B, 3, 3]
            acc_global: [B, T, 3]
        """
        # Pose stream
        _, h_pose = self.pose_gru(imu_seq)
        h_pose = h_pose[-1]
        q_pred = F.normalize(self.pose_fc(h_pose), dim=1, eps=1e-8)
        R_pred = quat_to_rotmat(q_pred)

        # Split IMU
        gyro_local = imu_seq[:, :, 0:3]
        acc_local = imu_seq[:, :, 3:6]

        # Differentiable projection to pseudo global frame
        R_t = R_pred.transpose(1, 2)
        acc_global = torch.matmul(acc_local, R_t)
        gyro_global = torch.matmul(gyro_local, R_t)

        # Heading stream
        imu_global = torch.cat([gyro_global, acc_global], dim=2)
        _, h_head = self.heading_gru(imu_global)
        h_head = h_head[-1]
        heading_pred = self.heading_fc(h_head)

        return heading_pred, R_pred, acc_global


def gravity_alignment_loss(R_pred, acc_local, gravity_sign=-1.0):
    """
    Physics-aware loss: align predicted global acceleration with gravity.
    R_pred: [B, 3, 3] (body -> world)
    acc_local: [B, T, 3]
    """
    R_t = R_pred.transpose(1, 2)
    acc_global = torch.matmul(acc_local, R_t)
    acc_mean = acc_global.mean(dim=1)
    g = torch.tensor([0.0, 0.0, gravity_sign], device=acc_local.device).view(1, 3)
    return F.mse_loss(acc_mean, g.expand_as(acc_mean))


# Example loss usage:
# heading_pred, R_pred, acc_global = model(imu_seq)
# loss_heading = F.mse_loss(heading_pred, heading_gt)
# loss_gravity = gravity_alignment_loss(R_pred, imu_seq[:, :, 3:6], gravity_sign=-1.0)
# loss = loss_heading + 0.1 * loss_gravity
