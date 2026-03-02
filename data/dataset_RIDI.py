import os
import numpy as np
from scipy.ndimage import gaussian_filter1d


def quat_conj(q):
    q = np.array(q, dtype=np.float32)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=np.float32)

def quat_to_rotmat(q):
    w, x, y, z = q
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
    return np.array([
        [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
    ], dtype=np.float32)


def moving_average(x, k):
    """简易滑动平均滤波，窗口 k>=1；k<=1 时原样返回。"""
    if k is None or k <= 1:
        return x
    k = int(k)
    kernel = np.ones(k, dtype=float) / float(k)
    if isinstance(x, np.ndarray) and x.ndim == 1:
        return np.convolve(x, kernel, mode="same")
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return np.stack([np.convolve(x[:, i], kernel, mode="same") for i in range(x.shape[1])], axis=1)
    return x


def _load_txt(path):
    with open(path, "r") as f:
        lines = [l for l in f if l.strip() and not l.startswith("#")]
    return np.loadtxt(lines)


def _interp_to(t_src, data, t_tgt):
    if t_src.ndim != 1:
        t_src = t_src.reshape(-1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    t_tgt = np.clip(t_tgt, t_src[0], t_src[-1])
    out = np.zeros((len(t_tgt), data.shape[1]), dtype=np.float32)
    for i in range(data.shape[1]):
        out[:, i] = np.interp(t_tgt, t_src, data[:, i])
    return out


ORI_ORDER = "wxyz"  # RIDI orientation.txt uses w, x, y, z
ORI_IS_W2B = False  # RIDI orientation is body->world
POSE_ORDER = "xyzw"  # RIDI pose.txt quaternions are x, y, z, w (Tango), swap to wxyz


def load_ridi_raw(seq_dir, acc_source="acce"):
    """
    加载 RIDI 原始数据：IMU(gyro/acc) 与 GT 位置/姿态。

    Args:
        seq_dir: RIDI 单条序列目录

    Returns:
        gyro_data: (N, 3)
        acc_data: (N, 3)
        pos_data: (N, 3)
        ori_data: (N, 4)
    """
    gyro_path = os.path.join(seq_dir, "gyro.txt")
    if acc_source == "acce":
        acc_path = os.path.join(seq_dir, "acce.txt")
    elif acc_source == "linacce":
        acc_path = os.path.join(seq_dir, "linacce.txt")
    else:
        raise ValueError(f"Unknown acc_source: {acc_source}")
    pose_path = os.path.join(seq_dir, "pose.txt")
    ori_path = os.path.join(seq_dir, "orientation.txt")

    gyro_arr = _load_txt(gyro_path)
    acc_arr = _load_txt(acc_path)
    pose_arr = _load_txt(pose_path)
    ori_arr = _load_txt(ori_path) if os.path.exists(ori_path) else None

    t_gyro = gyro_arr[:, 0]
    gyro = gyro_arr[:, 1:4]

    t_acc = acc_arr[:, 0]
    acc = _interp_to(t_acc, acc_arr[:, 1:4], t_gyro)

    t_pose = pose_arr[:, 0]
    pos = _interp_to(t_pose, pose_arr[:, 1:4], t_gyro)

    if pose_arr is not None and pose_arr.shape[1] >= 8:
        pose_quat = pose_arr[:, -4:]
        if POSE_ORDER == "xyzw":
            pose_quat = pose_quat[:, [3, 0, 1, 2]]
        ori = _interp_to(t_pose, pose_quat, t_gyro)
        # normalize quaternion to avoid drift from interpolation
        norm = np.linalg.norm(ori, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-8, None)
        ori = ori / norm
    elif ori_arr is not None and ori_arr.shape[1] >= 5:
        t_ori = ori_arr[:, 0]
        ori = _interp_to(t_ori, ori_arr[:, 1:5], t_gyro)
        if ORI_ORDER == "xyzw":
            ori = ori[:, [3, 0, 1, 2]]
        if ORI_IS_W2B:
            ori = np.stack([ori[:, 0], -ori[:, 1], -ori[:, 2], -ori[:, 3]], axis=1)
    else:
        ori = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(t_gyro), 1))

    return gyro.astype(np.float32), acc.astype(np.float32), pos.astype(np.float32), ori.astype(np.float32)


def window_dataset(gyro_data, acc_data, pos_data, ori_data,
                   window_size=160, stride=36, filter_window=20,
                   smooth_length=False, length_sigma=1.0,
                   return_ori=False, return_rel_ori=False,
                   return_delta_p=False, return_delta_p_world=False):
    m = min(gyro_data.shape[0], acc_data.shape[0], pos_data.shape[0], ori_data.shape[0])
    gyro_data = gyro_data[:m]
    acc_data = acc_data[:m]
    pos_data = pos_data[:m]
    ori_data = ori_data[:m]

    # Always compute XYZ deltas for 3D labels.
    pos_xyz = pos_data
    if filter_window and filter_window > 1:
        pos_xyz = moving_average(pos_xyz, filter_window)

    imu_gyro = []
    imu_acc = []
    y_len = []
    y_ori = []
    y_rel = []
    y_dp = []
    y_dp_world = []

    # Each window uses two midpoints:
    # start_idx = center - stride/2, end_idx = center + stride/2
    start_0 = window_size // 2 - stride // 2
    end_0 = window_size // 2 + stride // 2
    start_0 = max(0, min(start_0, len(pos_xyz) - 1))
    end_0 = max(0, min(end_0, len(pos_xyz) - 1))

    init_pos = pos_xyz[start_0, :2]
    init_head = 0.0

    max_start = gyro_data.shape[0] - window_size - 1
    for idx in range(0, max_start, stride):
        gyro_window = gyro_data[idx + 1: idx + 1 + window_size, :]
        acc_window = acc_data[idx + 1: idx + 1 + window_size, :]
        imu_gyro.append(gyro_window)
        imu_acc.append(acc_window)

        start_idx = idx + window_size // 2 - stride // 2
        end_idx = idx + window_size // 2 + stride // 2
        start_idx = max(0, min(start_idx, len(pos_xyz) - 1))
        end_idx = max(0, min(end_idx, len(pos_xyz) - 1))

        pos_start_xyz = pos_xyz[start_idx, :]
        pos_end_xyz = pos_xyz[end_idx, :]

        delta_len = np.linalg.norm(pos_end_xyz - pos_start_xyz)

        y_len.append(np.array([delta_len], dtype=np.float32))
        if return_delta_p:
            dp_world = (pos_end_xyz - pos_start_xyz).astype(np.float32)
            q_start = ori_data[start_idx].astype(np.float32)
            R_start = quat_to_rotmat(q_start)
            dp_body = (R_start.T @ dp_world.reshape(3, 1)).reshape(3,)
            y_dp.append(dp_body.astype(np.float32))
        if return_delta_p_world:
            if not return_delta_p:
                dp_world = (pos_end_xyz - pos_start_xyz).astype(np.float32)
            y_dp_world.append(dp_world.astype(np.float32))
        if return_ori:
            y_ori.append(ori_data[end_idx].astype(np.float32))
        if return_rel_ori:
            q_start = ori_data[start_idx].astype(np.float32)
            q_end = ori_data[end_idx].astype(np.float32)
            q_rel = quat_mul(quat_conj(q_start), q_end)
            y_rel.append(q_rel)

    x_gyro = np.array(imu_gyro)
    x_acc = np.array(imu_acc)
    y_len = np.array(y_len)
    if return_ori:
        y_ori = np.array(y_ori)
    if return_rel_ori:
        y_rel = np.array(y_rel)
    if return_delta_p:
        y_dp = np.array(y_dp)
    if return_delta_p_world:
        y_dp_world = np.array(y_dp_world)

    if smooth_length and len(y_len) > 0:
        y_len_smooth = gaussian_filter1d(y_len.flatten(), sigma=length_sigma)
        y_len = y_len_smooth.reshape(-1, 1)

    labels = [y_len]
    if return_ori:
        labels.append(y_ori)
    if return_rel_ori:
        labels.append(y_rel)
    if return_delta_p:
        labels.append(y_dp)
    if return_delta_p_world:
        labels.append(y_dp_world)
    return [x_gyro, x_acc], labels, init_pos, init_head
