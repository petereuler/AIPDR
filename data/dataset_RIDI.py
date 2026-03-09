import os

import numpy as np


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ],
        dtype=np.float32,
    )


def _load_txt(path):
    with open(path, "r") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    return np.loadtxt(lines)


def _interp_to(t_src, data, t_tgt):
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    t_tgt = np.clip(t_tgt, t_src[0], t_src[-1])
    out = np.zeros((len(t_tgt), data.shape[1]), dtype=np.float32)
    for i in range(data.shape[1]):
        out[:, i] = np.interp(t_tgt, t_src, data[:, i])
    return out


def load_ridi_raw(seq_dir, acc_source="acce"):
    gyro_arr = _load_txt(os.path.join(seq_dir, "gyro.txt"))
    acc_name = "acce.txt" if acc_source == "acce" else "linacce.txt"
    acc_arr = _load_txt(os.path.join(seq_dir, acc_name))
    pose_arr = _load_txt(os.path.join(seq_dir, "pose.txt"))
    ori_path = os.path.join(seq_dir, "orientation.txt")
    ori_arr = _load_txt(ori_path) if os.path.exists(ori_path) else None

    t_gyro = gyro_arr[:, 0]
    gyro = gyro_arr[:, 1:4]
    acc = _interp_to(acc_arr[:, 0], acc_arr[:, 1:4], t_gyro)
    pos = _interp_to(pose_arr[:, 0], pose_arr[:, 1:4], t_gyro)

    if pose_arr.shape[1] >= 8:
        pose_quat = pose_arr[:, -4:][:, [3, 0, 1, 2]]
        ori = _interp_to(pose_arr[:, 0], pose_quat, t_gyro)
        ori = ori / np.clip(np.linalg.norm(ori, axis=1, keepdims=True), 1e-8, None)
    elif ori_arr is not None and ori_arr.shape[1] >= 5:
        ori = _interp_to(ori_arr[:, 0], ori_arr[:, 1:5], t_gyro)
        ori = ori / np.clip(np.linalg.norm(ori, axis=1, keepdims=True), 1e-8, None)
    else:
        ori = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(t_gyro), 1))

    return gyro.astype(np.float32), acc.astype(np.float32), pos.astype(np.float32), ori.astype(np.float32)


def window_dataset(
    gyro_data,
    acc_data,
    pos_data,
    ori_data,
    window_size=160,
    stride=36,
    filter_window=0,
    smooth_length=False,
    length_sigma=1.0,
    return_ori=False,
    return_rel_ori=False,
    return_delta_p=False,
    return_delta_p_world=False,
):
    del filter_window, smooth_length, length_sigma

    m = min(gyro_data.shape[0], acc_data.shape[0], pos_data.shape[0], ori_data.shape[0])
    gyro_data = gyro_data[:m]
    acc_data = acc_data[:m]
    pos_data = pos_data[:m]
    ori_data = ori_data[:m]

    imu_gyro = []
    imu_acc = []
    y_len = []
    y_ori = []
    y_rel = []
    y_dp = []
    y_dp_world = []

    start_0 = max(0, min(window_size // 2 - stride // 2, len(pos_data) - 1))
    init_pos = pos_data[start_0, :2]
    init_head = 0.0

    max_start = gyro_data.shape[0] - window_size - 1
    for idx in range(0, max_start, stride):
        imu_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        imu_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])

        start_idx = max(0, min(idx + window_size // 2 - stride // 2, len(pos_data) - 1))
        end_idx = max(0, min(idx + window_size // 2 + stride // 2, len(pos_data) - 1))

        dp_world = (pos_data[end_idx] - pos_data[start_idx]).astype(np.float32)
        y_len.append(np.array([np.linalg.norm(dp_world)], dtype=np.float32))

        if return_delta_p:
            R_start = quat_to_rotmat(ori_data[start_idx].astype(np.float32))
            y_dp.append((R_start.T @ dp_world.reshape(3, 1)).reshape(3).astype(np.float32))
        if return_delta_p_world:
            y_dp_world.append(dp_world)
        if return_ori:
            y_ori.append(ori_data[end_idx].astype(np.float32))
        if return_rel_ori:
            y_rel.append(quat_mul(quat_conj(ori_data[start_idx]), ori_data[end_idx]))

    labels = [np.array(y_len)]
    if return_ori:
        labels.append(np.array(y_ori))
    if return_rel_ori:
        labels.append(np.array(y_rel))
    if return_delta_p:
        labels.append(np.array(y_dp))
    if return_delta_p_world:
        labels.append(np.array(y_dp_world))
    return [np.array(imu_gyro), np.array(imu_acc)], labels, init_pos, init_head
