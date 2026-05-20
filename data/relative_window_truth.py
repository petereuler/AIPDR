import os

import numpy as np

from data.dataset_OXIOD import get_oxiod_predefined_split_pairs, load_oxiod_raw
from data.dataset_RIDI import load_ridi_raw


def quat_normalize_np(q):
    q = np.asarray(q, dtype=np.float32)
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)


def quat_conj_np(q):
    q = np.asarray(q, dtype=np.float32)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul_np(q1, q2):
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    ).astype(np.float32)


def quat_to_rotmat_np(q):
    q = quat_normalize_np(q)
    w, x, y, z = q
    return np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ],
        dtype=np.float32,
    )


def canonicalize_quat_sequence_np(q_seq):
    q_seq = quat_normalize_np(q_seq)
    if q_seq.shape[0] == 0:
        return q_seq
    out = q_seq.copy()
    for idx in range(1, out.shape[0]):
        if np.dot(out[idx - 1], out[idx]) < 0.0:
            out[idx] *= -1.0
    return out


def _read_name_list(path):
    with open(path, "r") as f:
        return [line.strip().split(",")[0] for line in f if line.strip()]


def build_relative_pose_window(ori_window):
    q0_conj = quat_conj_np(ori_window[0:1])
    rel = quat_mul_np(np.repeat(q0_conj, ori_window.shape[0], axis=0), ori_window)
    rel[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return canonicalize_quat_sequence_np(rel)


def build_relative_disp_window(pos_window, ori_window):
    rel_world = (pos_window - pos_window[0:1]).astype(np.float32)
    r0 = quat_to_rotmat_np(ori_window[0].astype(np.float32))
    return (r0.T @ rel_world.T).T.astype(np.float32)


def build_step_disp_body_window(pos_window, ori_window):
    step_world = np.zeros_like(pos_window, dtype=np.float32)
    step_world[1:] = pos_window[1:] - pos_window[:-1]
    step_body = np.zeros_like(step_world, dtype=np.float32)
    for idx in range(1, len(pos_window)):
        r_prev = quat_to_rotmat_np(ori_window[idx - 1].astype(np.float32))
        step_body[idx] = (r_prev.T @ step_world[idx].reshape(3, 1)).reshape(3)
    return step_body.astype(np.float32)


def build_step_disp_world_window(pos_window):
    step_world = np.zeros_like(pos_window, dtype=np.float32)
    step_world[1:] = pos_window[1:] - pos_window[:-1]
    return step_world.astype(np.float32)


def extract_relative_windows(gyro_data, pos_data, ori_data, window_size=64, stride=64, start_offset=0):
    m = min(gyro_data.shape[0], pos_data.shape[0], ori_data.shape[0])
    gyro_data = gyro_data[:m]
    pos_data = pos_data[:m]
    ori_data = quat_normalize_np(ori_data[:m])

    pose_windows = []
    disp_windows = []
    max_start = gyro_data.shape[0] - window_size - 1
    start_offset = int(max(0, start_offset))
    for idx in range(start_offset, max_start, stride):
        start = idx + 1
        end = start + window_size
        pos_window = pos_data[start:end]
        ori_window = ori_data[start:end]
        if pos_window.shape[0] != window_size or ori_window.shape[0] != window_size:
            continue
        pose_windows.append(build_relative_pose_window(ori_window))
        disp_windows.append(build_relative_disp_window(pos_window, ori_window))

    if not pose_windows:
        pose = np.zeros((0, window_size, 4), dtype=np.float32)
        disp = np.zeros((0, window_size, 3), dtype=np.float32)
        return pose, disp
    return np.asarray(pose_windows, dtype=np.float32), np.asarray(disp_windows, dtype=np.float32)


def _load_sequences_from_ridi(ridi_root, names, window_size=64, stride=64, start_offset=0):
    data_root = os.path.join(ridi_root, "data")
    pose_sequences = []
    disp_sequences = []
    for name in names:
        seq_dir = os.path.join(data_root, name)
        if not os.path.isdir(seq_dir):
            continue
        gyro, _acc, pos_xyz, ori = load_ridi_raw(seq_dir, acc_source="linacce")
        pose, disp = extract_relative_windows(
            gyro,
            pos_xyz,
            ori,
            window_size=window_size,
            stride=stride,
            start_offset=start_offset,
        )
        if pose.shape[0] == 0:
            continue
        pose_sequences.append({"name": name, "truth": pose})
        disp_sequences.append({"name": name, "truth": disp})
    return pose_sequences, disp_sequences


def _load_sequences_from_oxiod(pairs, window_size=64, stride=64, start_offset=0):
    pose_sequences = []
    disp_sequences = []
    for name, imu_file, gt_file in pairs:
        gyro, _acc, pos_xyz, ori = load_oxiod_raw(imu_file, gt_file)
        pose, disp = extract_relative_windows(
            gyro,
            pos_xyz,
            ori,
            window_size=window_size,
            stride=stride,
            start_offset=start_offset,
        )
        if pose.shape[0] == 0:
            continue
        pose_sequences.append({"name": name, "truth": pose})
        disp_sequences.append({"name": name, "truth": disp})
    return pose_sequences, disp_sequences


def _extract_imu_windows(gyro_data, acc_data, window_size=64, stride=64, start_offset=0):
    m = min(gyro_data.shape[0], acc_data.shape[0])
    gyro_data = gyro_data[:m]
    acc_data = acc_data[:m]
    imu_windows = []
    max_start = gyro_data.shape[0] - window_size - 1
    start_offset = int(max(0, start_offset))
    for idx in range(start_offset, max_start, stride):
        start = idx + 1
        end = start + window_size
        gyro_window = gyro_data[start:end]
        acc_window = acc_data[start:end]
        if gyro_window.shape[0] != window_size or acc_window.shape[0] != window_size:
            continue
        imu_windows.append(np.concatenate([gyro_window, acc_window], axis=-1).astype(np.float32))
    if not imu_windows:
        return np.zeros((0, window_size, 6), dtype=np.float32)
    return np.asarray(imu_windows, dtype=np.float32)


def _extract_pose_imu_windows(gyro_data, acc_data, pos_data, ori_data, window_size=64, stride=64, start_offset=0):
    m = min(gyro_data.shape[0], acc_data.shape[0], pos_data.shape[0], ori_data.shape[0])
    gyro_data = gyro_data[:m]
    acc_data = acc_data[:m]
    pos_data = pos_data[:m]
    ori_data = quat_normalize_np(ori_data[:m])

    imu_windows = []
    pose_windows = []
    disp_windows = []
    step_disp_windows = []
    step_world_windows = []
    init_rot = None
    traj_rows = []
    pos_xy = np.zeros(2, dtype=np.float32)

    max_start = gyro_data.shape[0] - window_size - 1
    start_offset = int(max(0, start_offset))
    for idx in range(start_offset, max_start, stride):
        start = idx + 1
        end = start + window_size
        gyro_window = gyro_data[start:end]
        acc_window = acc_data[start:end]
        pos_window = pos_data[start:end]
        ori_window = ori_data[start:end]
        if (
            gyro_window.shape[0] != window_size
            or acc_window.shape[0] != window_size
            or pos_window.shape[0] != window_size
            or ori_window.shape[0] != window_size
        ):
            continue
        if init_rot is None:
            init_rot = quat_to_rotmat_np(ori_window[0].astype(np.float32))
        imu_windows.append(np.concatenate([gyro_window, acc_window], axis=-1).astype(np.float32))
        pose_windows.append(build_relative_pose_window(ori_window))
        disp_windows.append(build_relative_disp_window(pos_window, ori_window))
        step_world_window = build_step_disp_world_window(pos_window)
        step_disp_windows.append(build_step_disp_body_window(pos_window, ori_window))
        step_world_windows.append(step_world_window)
        for step_idx in range(1, len(step_world_window)):
            pos_xy = pos_xy + step_world_window[step_idx, :2]
        traj_rows.append(pos_xy.copy())

    if not imu_windows:
        return (
            np.zeros((0, window_size, 6), dtype=np.float32),
            np.zeros((0, window_size, 4), dtype=np.float32),
            np.zeros((0, window_size, 3), dtype=np.float32),
            np.zeros((0, window_size, 3), dtype=np.float32),
            np.zeros((0, window_size, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.eye(3, dtype=np.float32),
        )
    return (
        np.asarray(imu_windows, dtype=np.float32),
        np.asarray(pose_windows, dtype=np.float32),
        np.asarray(disp_windows, dtype=np.float32),
        np.asarray(step_disp_windows, dtype=np.float32),
        np.asarray(step_world_windows, dtype=np.float32),
        np.asarray(traj_rows, dtype=np.float32),
        init_rot.astype(np.float32),
    )


def _load_pose_imu_sequences_from_ridi(ridi_root, names, window_size=64, stride=64, start_offset=0):
    data_root = os.path.join(ridi_root, "data")
    sequences = []
    for name in names:
        seq_dir = os.path.join(data_root, name)
        if not os.path.isdir(seq_dir):
            continue
        gyro, acc, pos_xyz, ori = load_ridi_raw(seq_dir, acc_source="acce")
        imu, pose, disp, step_disp, step_world, gt_traj, init_rot = _extract_pose_imu_windows(
            gyro,
            acc,
            pos_xyz,
            ori,
            window_size=window_size,
            stride=stride,
            start_offset=start_offset,
        )
        if len(imu) == 0:
            continue
        sequences.append(
            {
                "name": name,
                "imu": imu,
                "truth": pose,
                "disp": disp,
                "step_disp": step_disp,
                "step_world": step_world,
                "gt_traj": gt_traj,
                "init_rot": init_rot,
            }
        )
    return sequences


def _load_pose_imu_sequences_from_oxiod(pairs, window_size=64, stride=64, start_offset=0):
    sequences = []
    for name, imu_file, gt_file in pairs:
        gyro, acc, pos_xyz, ori = load_oxiod_raw(imu_file, gt_file)
        imu, pose, disp, step_disp, step_world, gt_traj, init_rot = _extract_pose_imu_windows(
            gyro,
            acc,
            pos_xyz,
            ori,
            window_size=window_size,
            stride=stride,
            start_offset=start_offset,
        )
        if len(imu) == 0:
            continue
        sequences.append(
            {
                "name": name,
                "imu": imu,
                "truth": pose,
                "disp": disp,
                "step_disp": step_disp,
                "step_world": step_world,
                "gt_traj": gt_traj,
                "init_rot": init_rot,
            }
        )
    return sequences


def concatenate_sequence_field(sequences, field, window_size, feat_dim):
    if not sequences:
        return np.zeros((0, window_size, feat_dim), dtype=np.float32)
    return np.concatenate([seq[field] for seq in sequences], axis=0).astype(np.float32)


def concatenate_sequence_truth(sequences, window_size, feat_dim):
    return concatenate_sequence_field(sequences, "truth", window_size, feat_dim)


def concatenate_sequence_imu(sequences, window_size):
    if not sequences:
        return np.zeros((0, window_size, 6), dtype=np.float32)
    return np.concatenate([seq["imu"] for seq in sequences], axis=0).astype(np.float32)


def load_relative_truth_datasets(dataset_name, ridi_root, oxiod_root, window_size=64, stride=64, start_offset=0):
    dataset_name = dataset_name.upper()
    if dataset_name == "RIDI":
        data_root = os.path.join(ridi_root, "data")
        train_names = _read_name_list(os.path.join(data_root, "list_train_publish_v2.txt"))
        val_names = _read_name_list(os.path.join(data_root, "list_test_publish_v2.txt"))
        pose_train_seq, disp_train_seq = _load_sequences_from_ridi(
            ridi_root, train_names, window_size=window_size, stride=stride, start_offset=start_offset
        )
        pose_val_seq, disp_val_seq = _load_sequences_from_ridi(
            ridi_root, val_names, window_size=window_size, stride=stride, start_offset=start_offset
        )
    elif dataset_name == "OXIOD":
        train_pairs = get_oxiod_predefined_split_pairs(oxiod_root, split="train", sensor="syn")
        val_pairs = get_oxiod_predefined_split_pairs(oxiod_root, split="test", sensor="syn")
        pose_train_seq, disp_train_seq = _load_sequences_from_oxiod(
            train_pairs, window_size=window_size, stride=stride, start_offset=start_offset
        )
        pose_val_seq, disp_val_seq = _load_sequences_from_oxiod(
            val_pairs, window_size=window_size, stride=stride, start_offset=start_offset
        )
    else:
        raise ValueError(f"Unsupported DATASET={dataset_name}, expected RIDI or OXIOD")

    pose_train = concatenate_sequence_truth(pose_train_seq, window_size, 4)
    pose_val = concatenate_sequence_truth(pose_val_seq, window_size, 4)
    disp_train = concatenate_sequence_truth(disp_train_seq, window_size, 3)
    disp_val = concatenate_sequence_truth(disp_val_seq, window_size, 3)

    return {
        "pose_train": pose_train,
        "pose_val": pose_val,
        "disp_train": disp_train,
        "disp_val": disp_val,
        "pose_train_sequences": pose_train_seq,
        "pose_val_sequences": pose_val_seq,
        "disp_train_sequences": disp_train_seq,
        "disp_val_sequences": disp_val_seq,
    }


def load_pose_imu_relative_datasets(dataset_name, ridi_root, oxiod_root, window_size=64, stride=64, start_offset=0):
    dataset_name = dataset_name.upper()
    if dataset_name == "RIDI":
        data_root = os.path.join(ridi_root, "data")
        train_names = _read_name_list(os.path.join(data_root, "list_train_publish_v2.txt"))
        val_names = _read_name_list(os.path.join(data_root, "list_test_publish_v2.txt"))
        train_sequences = _load_pose_imu_sequences_from_ridi(
            ridi_root, train_names, window_size=window_size, stride=stride, start_offset=start_offset
        )
        val_sequences = _load_pose_imu_sequences_from_ridi(
            ridi_root, val_names, window_size=window_size, stride=stride, start_offset=start_offset
        )
    elif dataset_name == "OXIOD":
        train_pairs = get_oxiod_predefined_split_pairs(oxiod_root, split="train", sensor="syn")
        val_pairs = get_oxiod_predefined_split_pairs(oxiod_root, split="test", sensor="syn")
        train_sequences = _load_pose_imu_sequences_from_oxiod(
            train_pairs, window_size=window_size, stride=stride, start_offset=start_offset
        )
        val_sequences = _load_pose_imu_sequences_from_oxiod(
            val_pairs, window_size=window_size, stride=stride, start_offset=start_offset
        )
    else:
        raise ValueError(f"Unsupported DATASET={dataset_name}, expected RIDI or OXIOD")

    return {
        "imu_train": concatenate_sequence_imu(train_sequences, window_size),
        "imu_val": concatenate_sequence_imu(val_sequences, window_size),
        "pose_train": concatenate_sequence_field(train_sequences, "truth", window_size, 4),
        "pose_val": concatenate_sequence_field(val_sequences, "truth", window_size, 4),
        "disp_train": concatenate_sequence_field(train_sequences, "disp", window_size, 3),
        "disp_val": concatenate_sequence_field(val_sequences, "disp", window_size, 3),
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
    }
