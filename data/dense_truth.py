import os

import numpy as np

from data.dataset_OXIOD import (
    get_oxiod_predefined_split_pairs,
    load_oxiod_raw,
)
from data.dataset_RIDI import load_ridi_raw


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


def _read_name_list(path):
    with open(path, "r") as f:
        return [line.strip().split(",")[0] for line in f if line.strip()]


def _build_truth_window(pos_window, ori_window, truth_mode):
    rel_world = (pos_window - pos_window[0:1]).astype(np.float32)
    if truth_mode == "rel_pos_world":
        return rel_world

    r0 = quat_to_rotmat(ori_window[0].astype(np.float32))
    if truth_mode == "rel_pos_body":
        return (r0.T @ rel_world.T).T.astype(np.float32)

    if truth_mode == "step_pos_body":
        step_world = np.diff(pos_window, axis=0, prepend=pos_window[0:1]).astype(np.float32)
        return (r0.T @ step_world.T).T.astype(np.float32)

    raise ValueError(f"Unsupported truth_mode={truth_mode}")


def dense_truth_windows(
    gyro_data,
    pos_data,
    ori_data,
    window_size=64,
    stride=64,
    start_offset=0,
    truth_mode="rel_pos_body",
):
    m = min(gyro_data.shape[0], pos_data.shape[0], ori_data.shape[0])
    gyro_data = gyro_data[:m]
    pos_data = pos_data[:m]
    ori_data = ori_data[:m]

    windows = []
    max_start = gyro_data.shape[0] - window_size - 1
    start_offset = int(max(0, start_offset))
    for idx in range(start_offset, max_start, stride):
        start = idx + 1
        end = start + window_size
        pos_window = pos_data[start:end]
        ori_window = ori_data[start:end]
        if pos_window.shape[0] != window_size or ori_window.shape[0] != window_size:
            continue
        windows.append(_build_truth_window(pos_window, ori_window, truth_mode))

    if not windows:
        return np.zeros((0, window_size, 3), dtype=np.float32)
    return np.asarray(windows, dtype=np.float32)


def load_ridi_dense_truth_sequences(
    ridi_root,
    names,
    window_size=64,
    stride=64,
    start_offset=0,
    truth_mode="rel_pos_body",
):
    data_root = os.path.join(ridi_root, "data")
    sequences = []
    for name in names:
        seq_dir = os.path.join(data_root, name)
        if not os.path.isdir(seq_dir):
            continue
        gyro, _acc, pos_xyz, ori = load_ridi_raw(seq_dir, acc_source="linacce")
        truth = dense_truth_windows(
            gyro,
            pos_xyz,
            ori,
            window_size=window_size,
            stride=stride,
            start_offset=start_offset,
            truth_mode=truth_mode,
        )
        if truth.shape[0] > 0:
            sequences.append({"name": name, "truth": truth})
    return sequences


def load_oxiod_dense_truth_sequences(
    pairs,
    window_size=64,
    stride=64,
    start_offset=0,
    truth_mode="rel_pos_body",
):
    sequences = []
    for name, imu_file, gt_file in pairs:
        gyro, _acc, pos_xyz, ori = load_oxiod_raw(imu_file, gt_file)
        truth = dense_truth_windows(
            gyro,
            pos_xyz,
            ori,
            window_size=window_size,
            stride=stride,
            start_offset=start_offset,
            truth_mode=truth_mode,
        )
        if truth.shape[0] > 0:
            sequences.append({"name": name, "truth": truth})
    return sequences


def concatenate_truth(sequences, window_size=64):
    if not sequences:
        return np.zeros((0, window_size, 3), dtype=np.float32)
    return np.concatenate([seq["truth"] for seq in sequences], axis=0).astype(np.float32)


def load_dense_truth_dataset(
    dataset_name,
    ridi_root,
    oxiod_root,
    window_size=64,
    stride=64,
    start_offset=0,
    truth_mode="rel_pos_body",
):
    dataset_name = dataset_name.upper()
    if dataset_name == "RIDI":
        data_root = os.path.join(ridi_root, "data")
        train_names = _read_name_list(os.path.join(data_root, "list_train_publish_v2.txt"))
        val_names = _read_name_list(os.path.join(data_root, "list_test_publish_v2.txt"))
        train_sequences = load_ridi_dense_truth_sequences(
            ridi_root,
            train_names,
            window_size=window_size,
            stride=stride,
            start_offset=start_offset,
            truth_mode=truth_mode,
        )
        val_sequences = load_ridi_dense_truth_sequences(
            ridi_root,
            val_names,
            window_size=window_size,
            stride=stride,
            start_offset=start_offset,
            truth_mode=truth_mode,
        )
    elif dataset_name == "OXIOD":
        train_pairs = get_oxiod_predefined_split_pairs(oxiod_root, split="train", sensor="syn")
        val_pairs = get_oxiod_predefined_split_pairs(oxiod_root, split="test", sensor="syn")
        train_sequences = load_oxiod_dense_truth_sequences(
            train_pairs,
            window_size=window_size,
            stride=stride,
            start_offset=start_offset,
            truth_mode=truth_mode,
        )
        val_sequences = load_oxiod_dense_truth_sequences(
            val_pairs,
            window_size=window_size,
            stride=stride,
            start_offset=start_offset,
            truth_mode=truth_mode,
        )
    else:
        raise ValueError(f"Unsupported DATASET={dataset_name}, expected RIDI or OXIOD")

    return (
        concatenate_truth(train_sequences, window_size=window_size),
        concatenate_truth(val_sequences, window_size=window_size),
        train_sequences,
        val_sequences,
    )
