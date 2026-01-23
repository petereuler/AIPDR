import numpy as np
import torch


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


def compute_init_rot(ori: np.ndarray, pos3d: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Compute init rotation aligned to chord frame for each window."""
    pos2d = pos3d[:, :2]
    max_start = pos2d.shape[0] - window_size - 1
    if max_start <= 0:
        return np.zeros((0, 3, 3), dtype=np.float32)

    a_indices = []
    b_indices = []
    for idx in range(0, max_start, stride):
        a = idx + window_size // 2 - stride // 2
        b = idx + window_size // 2 + stride // 2
        a = max(0, min(a, len(pos2d) - 1))
        b = max(0, min(b, len(pos2d) - 1))
        a_indices.append(a)
        b_indices.append(b)
    a_indices = np.array(a_indices, dtype=np.int64)
    b_indices = np.array(b_indices, dtype=np.int64)

    if b_indices.size == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)

    a0 = a_indices[0]
    b0 = b_indices[0]
    diff_0 = pos2d[b0] - pos2d[a0]
    init_head = float(np.arctan2(diff_0[1], diff_0[0]))

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
    yaw_b2w = np.arctan2(Rq[1, 0], Rq[0, 0])
    diff = init_head - yaw_b2w
    yaw_offset = np.arctan2(np.sin(diff), np.cos(diff))
    cz = np.cos(yaw_offset)
    sz = np.sin(yaw_offset)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    init_rot = Rz @ Rq
    init_rot = np.repeat(init_rot[None, :, :], len(b_indices), axis=0)
    return init_rot
