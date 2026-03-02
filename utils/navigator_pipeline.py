import numpy as np
import torch

from models.pose_net import quat_conj, quat_mul, quat_to_rotmat


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def accumulate_rotations(R_delta: torch.Tensor, seq_id: torch.Tensor, init_rot: torch.Tensor) -> torch.Tensor:
    """Accumulate relative rotations into absolute rotations per sequence."""
    R_abs = []
    prev_seq = None
    current = None
    for i in range(R_delta.size(0)):
        sid = int(seq_id[i].item())
        if prev_seq is None or sid != prev_seq:
            current = init_rot[i]
        current = torch.matmul(current, R_delta[i])
        R_abs.append(current)
        prev_seq = sid
    return torch.stack(R_abs, dim=0)


def compute_init_rot(ori: np.ndarray, pos_xyz: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Compute init rotation using pure GT orientation (no heading alignment)."""
    max_start = pos_xyz.shape[0] - window_size - 1
    if max_start <= 0:
        return np.zeros((0, 3, 3), dtype=np.float32)

    b_indices = []
    for idx in range(0, max_start, stride):
        b = idx + window_size // 2 + stride // 2
        b = max(0, min(b, len(ori) - 1))
        b_indices.append(b)
    b_indices = np.array(b_indices, dtype=np.int64)
    if b_indices.size == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)

    init_q = ori[b_indices[0]].astype(np.float32)
    iw, ix, iy, iz = init_q
    ww = iw * iw
    xx = ix * ix
    yy = iy * iy
    zz = iz * iz
    wx = iw * ix
    wy = iw * iy
    wz = iw * iz
    xy = ix * iy
    xz = ix * iz
    yz = iy * iz
    Rq = np.array([
        [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
    ], dtype=np.float32)
    init_rot = np.repeat(Rq[None, :, :], len(b_indices), axis=0)
    return init_rot


def integrate_gyro_quat(gyro: torch.Tensor, q_start: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Integrate body-frame angular velocity into a quaternion sequence.
    gyro: [B, T, 3] (rad/s)
    q_start: [B, 4] (w, x, y, z)
    returns: q_seq [B, T, 4]
    """
    bsz, tlen, _ = gyro.shape
    q_seq = []
    q_curr = q_start
    for t in range(tlen):
        w = gyro[:, t, :]
        w_norm = torch.norm(w, dim=1, keepdim=True) + 1e-8
        angle = w_norm * dt
        axis = w / w_norm
        half = 0.5 * angle
        dq = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=1)
        q_curr = quat_mul(q_curr, dq)
        q_seq.append(q_curr)
    return torch.stack(q_seq, dim=1)


def slerp_identity_to(q_target: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    SLERP from identity to q_target.
    q_target: [B, 4]
    alpha: [B, T] in [0,1]
    returns: [B, T, 4]
    """
    if q_target.dim() == 3:
        q_target = q_target[:, -1, :]
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(0)
    if q_target.dim() != 2 or q_target.size(1) != 4:
        raise ValueError(f"slerp_identity_to: q_target shape {tuple(q_target.shape)}")
    if alpha.dim() != 2:
        raise ValueError(f"slerp_identity_to: alpha shape {tuple(alpha.shape)}")
    q = q_target / (q_target.norm(dim=1, keepdim=True) + 1e-8)
    dot = q[:, 0:1]
    sign = torch.where(dot < 0.0, -1.0, 1.0)
    q = q * sign

    dot = q[:, 0:1].clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp(min=1e-8)

    # alpha: [1, T] or [B, T], theta/sin_theta: [B, 1]
    w0 = torch.sin((1.0 - alpha) * theta) / sin_theta
    w1 = torch.sin(alpha * theta) / sin_theta
    w0 = w0.unsqueeze(-1)
    w1 = w1.unsqueeze(-1)

    q0 = torch.zeros_like(q)
    q0[:, 0] = 1.0
    q0 = q0.unsqueeze(1)
    q1 = q.unsqueeze(1)
    return w0 * q0 + w1 * q1


def gyro_slerp_align(gyro: torch.Tensor, q_start: torch.Tensor, q_end: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Gyro-guided integration with end-point correction via SLERP.
    gyro: [B, T, 3]
    q_start: [B, 4]
    q_end: [B, 4]
    returns: q_final_seq [B, T, 4]
    """
    if q_start.dim() == 3:
        q_start = q_start[:, 0, :]
    if q_end.dim() == 3:
        q_end = q_end[:, -1, :]
    if q_start.size(0) != gyro.size(0) or q_end.size(0) != gyro.size(0):
        raise ValueError(
            f"gyro_slerp_align batch mismatch: gyro {tuple(gyro.shape)}, "
            f"q_start {tuple(q_start.shape)}, q_end {tuple(q_end.shape)}"
        )
    q_gyro = integrate_gyro_quat(gyro, q_start, dt)
    q_end_pred = q_gyro[:, -1, :]
    q_err = quat_mul(q_end, quat_conj(q_end_pred))
    alpha = torch.linspace(0.0, 1.0, q_gyro.size(1), device=gyro.device).unsqueeze(0)
    q_corr = slerp_identity_to(q_err, alpha)
    q_corr_flat = q_corr.reshape(-1, 4)
    q_gyro_flat = q_gyro.reshape(-1, 4)
    q_final = quat_mul(q_corr_flat, q_gyro_flat).reshape_as(q_gyro)
    return q_final
