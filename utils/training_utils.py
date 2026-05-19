import os

import numpy as np
import torch

from data.dataset_OXIOD import (
    get_oxiod_predefined_split_pairs,
    load_oxiod_raw,
    window_dataset as oxiod_window,
)
from data.dataset_RIDI import load_ridi_raw, window_dataset as ridi_window


def _quat_conj_batch(q):
    q = np.asarray(q, dtype=np.float32).reshape(-1, 4)
    return np.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], axis=1)


def _quat_mul_batch(q1, q2):
    q1 = np.asarray(q1, dtype=np.float32).reshape(-1, 4)
    q2 = np.asarray(q2, dtype=np.float32).reshape(-1, 4)
    if q1.shape[0] == 1 and q2.shape[0] > 1:
        q1 = np.repeat(q1, q2.shape[0], axis=0)
    if q2.shape[0] == 1 and q1.shape[0] > 1:
        q2 = np.repeat(q2, q1.shape[0], axis=0)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def _quat_to_rotmat(q):
    w, x, y, z = q
    return np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ],
        dtype=np.float32,
    )


def _tensorize(arrays, device):
    return [torch.tensor(array, dtype=torch.float32, device=device) for array in arrays]


def _pack_absheading_outputs(
    xg_tr,
    xa_tr,
    yl_tr,
    yp_tr,
    ypw_tr,
    yo_tr,
    yr_tr,
    yi_tr,
    yalign_tr,
    xg_va,
    xa_va,
    yl_va,
    yp_va,
    ypw_va,
    yo_va,
    yr_va,
    yi_va,
    yalign_va,
    device,
    return_delta_p,
    return_delta_p_world,
    return_ori,
    return_rel_ori,
    return_init,
    return_align,
    align_init_quat,
):
    x_tr, x_va = _tensorize(
        [
            np.concatenate([np.concatenate(xg_tr, axis=0), np.concatenate(xa_tr, axis=0)], axis=-1),
            np.concatenate([np.concatenate(xg_va, axis=0), np.concatenate(xa_va, axis=0)], axis=-1),
        ],
        device,
    )
    ylen_tr, ylen_va = _tensorize([np.concatenate(yl_tr, axis=0), np.concatenate(yl_va, axis=0)], device)

    outputs = [x_tr, ylen_tr]
    if return_delta_p:
        outputs.append(torch.tensor(np.concatenate(yp_tr, axis=0), dtype=torch.float32, device=device))
    if return_delta_p_world:
        outputs.append(torch.tensor(np.concatenate(ypw_tr, axis=0), dtype=torch.float32, device=device))
    if return_ori:
        outputs.append(torch.tensor(np.concatenate(yo_tr, axis=0), dtype=torch.float32, device=device))
    if return_rel_ori:
        outputs.append(torch.tensor(np.concatenate(yr_tr, axis=0), dtype=torch.float32, device=device))
    if return_init:
        outputs.append(torch.tensor(np.concatenate(yi_tr, axis=0), dtype=torch.float32, device=device))
    if return_align and align_init_quat:
        outputs.append(torch.tensor(np.concatenate(yalign_tr, axis=0), dtype=torch.float32, device=device))

    outputs += [x_va, ylen_va]
    if return_delta_p:
        outputs.append(torch.tensor(np.concatenate(yp_va, axis=0), dtype=torch.float32, device=device))
    if return_delta_p_world:
        outputs.append(torch.tensor(np.concatenate(ypw_va, axis=0), dtype=torch.float32, device=device))
    if return_ori:
        outputs.append(torch.tensor(np.concatenate(yo_va, axis=0), dtype=torch.float32, device=device))
    if return_rel_ori:
        outputs.append(torch.tensor(np.concatenate(yr_va, axis=0), dtype=torch.float32, device=device))
    if return_init:
        outputs.append(torch.tensor(np.concatenate(yi_va, axis=0), dtype=torch.float32, device=device))
    if return_align and align_init_quat:
        outputs.append(torch.tensor(np.concatenate(yalign_va, axis=0), dtype=torch.float32, device=device))

    return tuple(outputs)


def load_data_ridi_absheading(
    ridi_root,
    device,
    window_size=160,
    stride=32,
    start_offset=0,
    return_ori=False,
    return_rel_ori=False,
    return_seq=False,
    return_init=False,
    return_delta_p=False,
    return_delta_p_world=False,
    align_init_quat=False,
    acc_source="acce",
    align_init_quat_to_labels=True,
    return_align=False,
):
    del return_seq

    data_root = os.path.join(ridi_root, "data")
    train_list = os.path.join(data_root, "list_train_publish_v2.txt")
    test_list = os.path.join(data_root, "list_test_publish_v2.txt")

    def _load_list(path):
        with open(path, "r") as f:
            return [line.strip().split(",")[0] for line in f if line.strip()]

    train_names = _load_list(train_list)
    val_names = _load_list(test_list)

    xg_tr, xa_tr, yl_tr, yp_tr, ypw_tr, yo_tr, yr_tr, yi_tr, yalign_tr = [], [], [], [], [], [], [], [], []
    xg_va, xa_va, yl_va, yp_va, ypw_va, yo_va, yr_va, yi_va, yalign_va = [], [], [], [], [], [], [], [], []

    for name in train_names + val_names:
        seq_dir = os.path.join(data_root, name)
        if not os.path.isdir(seq_dir):
            continue
        gyro, acc, pos_xyz, ori = load_ridi_raw(seq_dir, acc_source=acc_source)
        q_align = _quat_conj_batch(ori[0])[0] if align_init_quat else None
        R_align = _quat_to_rotmat(q_align) if align_init_quat else None

        max_start = gyro.shape[0] - window_size - 1
        a_indices = []
        for idx in range(0, max_start, stride):
            a = idx + window_size // 2 - stride // 2
            a = max(0, min(a, len(pos_xyz) - 1))
            a_indices.append(a)
        if not a_indices:
            continue

        [gx, ax], labels, _, _ = ridi_window(
            gyro,
            acc,
            pos_xyz,
            ori,
            window_size=window_size,
            stride=stride,
            start_offset=start_offset,
            filter_window=0,
            smooth_length=False,
            return_ori=return_ori,
            return_rel_ori=return_rel_ori,
            return_delta_p=return_delta_p,
            return_delta_p_world=return_delta_p_world,
        )
        if gx.shape[0] == 0:
            continue

        label_idx = 1
        dori = labels[label_idx] if return_ori else None
        label_idx += int(return_ori)
        drel = labels[label_idx] if return_rel_ori else None
        label_idx += int(return_rel_ori)
        dp = labels[label_idx] if return_delta_p else None
        label_idx += int(return_delta_p)
        dp_world = labels[label_idx] if return_delta_p_world else None

        if align_init_quat_to_labels and align_init_quat and return_ori:
            dori = _quat_mul_batch(q_align, dori)

        init_rot = _quat_to_rotmat(ori[a_indices[0]].astype(np.float32))
        if align_init_quat_to_labels and align_init_quat and return_init:
            init_rot = R_align @ init_rot

        target = (xg_va, xa_va, yl_va, yp_va, ypw_va, yo_va, yr_va, yi_va, yalign_va) if name in val_names else (
            xg_tr, xa_tr, yl_tr, yp_tr, ypw_tr, yo_tr, yr_tr, yi_tr, yalign_tr
        )
        target[0].append(gx)
        target[1].append(ax)
        target[2].append(labels[0])
        if return_delta_p:
            target[3].append(dp)
        if return_delta_p_world:
            target[4].append(dp_world)
        if return_ori:
            target[5].append(dori)
        if return_rel_ori:
            target[6].append(drel)
        if return_init:
            target[7].append(np.repeat(init_rot[None, :, :], gx.shape[0], axis=0))
        if return_align and align_init_quat:
            target[8].append(np.repeat(R_align[None, :, :], gx.shape[0], axis=0))

    return _pack_absheading_outputs(
        xg_tr, xa_tr, yl_tr, yp_tr, ypw_tr, yo_tr, yr_tr, yi_tr, yalign_tr,
        xg_va, xa_va, yl_va, yp_va, ypw_va, yo_va, yr_va, yi_va, yalign_va,
        device,
        return_delta_p,
        return_delta_p_world,
        return_ori,
        return_rel_ori,
        return_init,
        return_align,
        align_init_quat,
    )


def load_data_oxiod_absheading(
    oxiod_root,
    device,
    window_size=160,
    stride=32,
    start_offset=0,
    return_ori=False,
    return_rel_ori=False,
    return_seq=False,
    return_init=False,
    return_delta_p=False,
    return_delta_p_world=False,
    align_init_quat=False,
    align_init_quat_to_labels=True,
    return_align=False,
):
    del return_seq

    train_pairs = get_oxiod_predefined_split_pairs(oxiod_root, split="train", sensor="syn")
    val_pairs = get_oxiod_predefined_split_pairs(oxiod_root, split="test", sensor="syn")

    xg_tr, xa_tr, yl_tr, yp_tr, ypw_tr, yo_tr, yr_tr, yi_tr, yalign_tr = [], [], [], [], [], [], [], [], []
    xg_va, xa_va, yl_va, yp_va, ypw_va, yo_va, yr_va, yi_va, yalign_va = [], [], [], [], [], [], [], [], []

    for seq_idx, (_name, imu, gt) in enumerate(train_pairs + val_pairs):
        gyro, acc, pos_xyz, ori = load_oxiod_raw(imu, gt)
        q_align = _quat_conj_batch(ori[0])[0] if align_init_quat else None
        R_align = _quat_to_rotmat(q_align) if align_init_quat else None

        max_start = gyro.shape[0] - window_size - 1
        a_indices = []
        for idx in range(0, max_start, stride):
            a = idx + window_size // 2 - stride // 2
            a = max(0, min(a, len(pos_xyz) - 1))
            a_indices.append(a)
        if not a_indices:
            continue

        [gx, ax], labels, _, _ = oxiod_window(
            gyro,
            acc,
            pos_xyz,
            ori,
            window_size=window_size,
            stride=stride,
            start_offset=start_offset,
            filter_window=0,
            smooth_length=False,
            return_ori=return_ori,
            return_rel_ori=return_rel_ori,
            return_delta_p=return_delta_p,
            return_delta_p_world=return_delta_p_world,
        )
        if gx.shape[0] == 0:
            continue

        label_idx = 1
        dori = labels[label_idx] if return_ori else None
        label_idx += int(return_ori)
        drel = labels[label_idx] if return_rel_ori else None
        label_idx += int(return_rel_ori)
        dp = labels[label_idx] if return_delta_p else None
        label_idx += int(return_delta_p)
        dp_world = labels[label_idx] if return_delta_p_world else None

        if align_init_quat_to_labels and align_init_quat and return_ori:
            dori = _quat_mul_batch(q_align, dori)

        init_rot = _quat_to_rotmat(ori[a_indices[0]].astype(np.float32))
        if align_init_quat_to_labels and align_init_quat and return_init:
            init_rot = R_align @ init_rot

        is_val = seq_idx >= len(train_pairs)
        target = (xg_va, xa_va, yl_va, yp_va, ypw_va, yo_va, yr_va, yi_va, yalign_va) if is_val else (
            xg_tr, xa_tr, yl_tr, yp_tr, ypw_tr, yo_tr, yr_tr, yi_tr, yalign_tr
        )
        target[0].append(gx)
        target[1].append(ax)
        target[2].append(labels[0])
        if return_delta_p:
            target[3].append(dp)
        if return_delta_p_world:
            target[4].append(dp_world)
        if return_ori:
            target[5].append(dori)
        if return_rel_ori:
            target[6].append(drel)
        if return_init:
            target[7].append(np.repeat(init_rot[None, :, :], gx.shape[0], axis=0))
        if return_align and align_init_quat:
            target[8].append(np.repeat(R_align[None, :, :], gx.shape[0], axis=0))

    return _pack_absheading_outputs(
        xg_tr, xa_tr, yl_tr, yp_tr, ypw_tr, yo_tr, yr_tr, yi_tr, yalign_tr,
        xg_va, xa_va, yl_va, yp_va, ypw_va, yo_va, yr_va, yi_va, yalign_va,
        device,
        return_delta_p,
        return_delta_p_world,
        return_ori,
        return_rel_ori,
        return_init,
        return_align,
        align_init_quat,
    )
