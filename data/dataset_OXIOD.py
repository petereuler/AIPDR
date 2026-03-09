import os

import numpy as np
import pandas as pd


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


def load_oxiod_raw(imu_data_filename, gt_data_filename, trim_head=1200, trim_tail=300):
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    if trim_head > 0:
        imu_data = imu_data[trim_head:]
        gt_data = gt_data[trim_head:]
    if trim_tail > 0:
        imu_data = imu_data[:-trim_tail]
        gt_data = gt_data[:-trim_tail]

    m = min(len(imu_data), len(gt_data))
    imu_data = imu_data[:m]
    gt_data = gt_data[:m]

    gyro = imu_data[:, 4:7].astype(np.float32)
    acc = imu_data[:, 10:13].astype(np.float32)
    pos = gt_data[:, 2:5].astype(np.float32)
    ori = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1).astype(np.float32)
    ori = ori / np.clip(np.linalg.norm(ori, axis=1, keepdims=True), 1e-8, None)
    return gyro, acc, pos, ori


def get_oxiod_predefined_split_pairs(oxiod_root, split="train", sensor="syn"):
    split = split.lower()
    predefined_files = [
        os.path.join("handheld", "data1", "imu1.csv"),
        os.path.join("handheld", "data1", "imu3.csv"),
        os.path.join("handheld", "data1", "imu4.csv"),
        os.path.join("handheld", "data1", "imu7.csv"),
        os.path.join("handheld", "data2", "imu1.csv"),
        os.path.join("handheld", "data2", "imu2.csv"),
        os.path.join("handheld", "data2", "imu3.csv"),
        os.path.join("handheld", "data3", "imu2.csv"),
        os.path.join("handheld", "data3", "imu3.csv"),
        os.path.join("handheld", "data3", "imu4.csv"),
        os.path.join("handheld", "data3", "imu5.csv"),
        os.path.join("handheld", "data4", "imu2.csv"),
        os.path.join("handheld", "data4", "imu4.csv"),
        os.path.join("handheld", "data4", "imu5.csv"),
        os.path.join("handheld", "data5", "imu1.csv"),
        os.path.join("handheld", "data5", "imu2.csv"),
        os.path.join("handheld", "data5", "imu4.csv"),
    ]
    predefined_test = {
        os.path.join("handheld", "data1", "imu4.csv"),
        os.path.join("handheld", "data2", "imu2.csv"),
        os.path.join("handheld", "data3", "imu4.csv"),
        os.path.join("handheld", "data4", "imu5.csv"),
        os.path.join("handheld", "data5", "imu1.csv"),
    }

    pairs = []
    for rel_file in predefined_files:
        is_test = rel_file in predefined_test
        if (split == "train" and is_test) or (split == "test" and not is_test):
            continue
        rel_dir = os.path.dirname(rel_file)
        imu_name = os.path.basename(rel_file)
        imu_path = os.path.join(oxiod_root, rel_dir, sensor, imu_name)
        gt_path = os.path.join(oxiod_root, rel_dir, sensor, imu_name.replace("imu", "vi"))
        if os.path.exists(imu_path) and os.path.exists(gt_path):
            pairs.append((rel_file.replace(".csv", ""), imu_path, gt_path))
    return pairs


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
