"""
训练相关的工具函数
包含数据加载、数据增强、损失函数和可视化函数
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data.dataset_OXIOD import load_oxiod_raw, window_dataset as oxiod_window
from data.dataset_SELFMADE import load_selfmade_raw, window_dataset as selfmade_window
from data.dataset_RONIN import load_ronin_raw, window_dataset as ronin_window
from data.dataset_RIDI import load_ridi_raw, window_dataset as ridi_window


# ======= 损失函数 =======
def len_loss(pred, target):
    """步长回归损失函数"""
    return F.mse_loss(pred, target)


# ======= 数据加载函数 =======
def load_data_2d_oxiod(data_root, device, window_size=160, stride=32):
    """
    加载 OXIOD 数据集并分割为训练集和验证集
    
    Args:
        data_root: OXIOD 数据集根目录
        device: torch 设备
        window_size: 窗口大小
        stride: 步长
    
    Returns:
        x_tr, ylen_tr, yhead_tr, x_va, ylen_va, yhead_va
    """
    imu_files = [
        os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu1.csv'),
        os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu3.csv'),
        os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu4.csv'),
        os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu7.csv'),
        os.path.join(data_root, 'handheld', 'data2', 'syn', 'imu1.csv'),
        os.path.join(data_root, 'handheld', 'data2', 'syn', 'imu2.csv'),
        os.path.join(data_root, 'handheld', 'data2', 'syn', 'imu3.csv'),
        os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu2.csv'),
        os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu3.csv'),
        os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu4.csv'),
        os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu5.csv'),
        os.path.join(data_root, 'handheld', 'data4', 'syn', 'imu2.csv'),
        os.path.join(data_root, 'handheld', 'data4', 'syn', 'imu4.csv'),
        os.path.join(data_root, 'handheld', 'data4', 'syn', 'imu5.csv'),
        os.path.join(data_root, 'handheld', 'data5', 'syn', 'imu1.csv'),
        os.path.join(data_root, 'handheld', 'data5', 'syn', 'imu2.csv'),
        os.path.join(data_root, 'handheld', 'data5', 'syn', 'imu4.csv'),
    ]
    gt_files = [f.replace("imu", "vi") for f in imu_files]

    val_set = set([
        os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu4.csv'),
        os.path.join(data_root, 'handheld', 'data2', 'syn', 'imu2.csv'),
        os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu4.csv'),
        os.path.join(data_root, 'handheld', 'data4', 'syn', 'imu5.csv'),
        os.path.join(data_root, 'handheld', 'data5', 'syn', 'imu1.csv'),
    ])

    xg_tr, xa_tr, yl_tr, yh_tr = [], [], [], []
    xg_va, xa_va, yl_va, yh_va = [], [], [], []
    
    for imu, gt in zip(imu_files, gt_files):
        gyro, acc, pos3d, ori = load_oxiod_raw(imu, gt)
        
        [gx, ax], [dl, dh], _, _ = oxiod_window(
            gyro, acc, pos3d, ori,
            mode="2d",
            window_size=window_size,
            stride=stride,
            filter_window=20,
            smooth_heading=True,  # 启用航向角平滑，提高真值轨迹光滑性
            heading_sigma=1.5,    # 航向角高斯平滑标准差
            smooth_length=False,   # 不平滑步长，只平滑航向
            length_sigma=1.0,    # 步长高斯平滑标准差
        )
        if imu in val_set:
            xg_va.append(gx)
            xa_va.append(ax)
            yl_va.append(dl)
            yh_va.append(dh)
        else:
            xg_tr.append(gx)
            xa_tr.append(ax)
            yl_tr.append(dl)
            yh_tr.append(dh)

    x_tr = np.concatenate(xg_tr, axis=0)
    x_tr = np.concatenate([x_tr, np.concatenate(xa_tr, axis=0)], axis=-1)
    x_tr = torch.tensor(x_tr, dtype=torch.float32, device=device)
    ylen_tr = torch.tensor(np.concatenate(yl_tr, axis=0), dtype=torch.float32, device=device)
    yhead_tr = torch.tensor(np.concatenate(yh_tr, axis=0), dtype=torch.float32, device=device)

    x_va = np.concatenate(xg_va, axis=0)
    x_va = np.concatenate([x_va, np.concatenate(xa_va, axis=0)], axis=-1)
    x_va = torch.tensor(x_va, dtype=torch.float32, device=device)
    ylen_va = torch.tensor(np.concatenate(yl_va, axis=0), dtype=torch.float32, device=device)
    yhead_va = torch.tensor(np.concatenate(yh_va, axis=0), dtype=torch.float32, device=device)

    return x_tr, ylen_tr, yhead_tr, x_va, ylen_va, yhead_va


def load_data_2d_selfmade(selfmade_root, device, window_size=160, stride=32):
    """
    加载 SELFMADE 数据集并分割为训练集和验证集
    
    Args:
        selfmade_root: SELFMADE 数据集根目录
        device: torch 设备
        window_size: 窗口大小
        stride: 步长
    
    Returns:
        x_tr, ylen_tr, yhead_tr, x_va, ylen_va, yhead_va
    """
    files = []
    for r, d, fns in os.walk(selfmade_root):
        for fn in fns:
            if fn.lower().endswith('.csv') or fn.lower().endswith('.mat'):
                files.append(os.path.join(r, fn))
    limit = os.getenv('SELFMADE_LIMIT', None)
    if limit is not None:
        try:
            k = int(limit)
            if k > 0:
                files = files[:k]
        except Exception:
            pass
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError("No SELFMADE CSV files found")
    n = len(files)
    split = max(1, int(0.2 * n))
    val_set = set(files[-split:])
    xg_tr, xa_tr, yl_tr, yh_tr = [], [], [], []
    xg_va, xa_va, yl_va, yh_va = [], [], [], []
    for fp in files:
        gyro, acc, pos3d, ori = load_selfmade_raw(fp)
        
        [gx, ax], [dl, dh], _, _ = selfmade_window(
            gyro, acc, pos3d, ori,
            mode="2d",
            window_size=window_size,
            stride=stride,
            filter_window=10,
            smooth_heading=True,  # 启用航向角平滑，提高真值轨迹光滑性
            heading_sigma=1.5,    # 航向角高斯平滑标准差
            smooth_length=False,   # 不平滑步长，只平滑航向
            length_sigma=1.0,    # 步长高斯平滑标准差
        )
        if gx.shape[0] == 0:
            continue
        if fp in val_set:
            xg_va.append(gx)
            xa_va.append(ax)
            yl_va.append(dl)
            yh_va.append(dh)
        else:
            xg_tr.append(gx)
            xa_tr.append(ax)
            yl_tr.append(dl)
            yh_tr.append(dh)
    
    if len(xg_tr) == 0:
        raise RuntimeError("Training set is empty!")
    if len(xg_va) == 0:
        print("Warning: Validation set is empty!")
        
    x_tr = np.concatenate(xg_tr, axis=0)
    x_tr = np.concatenate([x_tr, np.concatenate(xa_tr, axis=0)], axis=-1)
    x_tr = torch.tensor(x_tr, dtype=torch.float32, device=device)
    ylen_tr = torch.tensor(np.concatenate(yl_tr, axis=0), dtype=torch.float32, device=device)
    yhead_tr = torch.tensor(np.concatenate(yh_tr, axis=0), dtype=torch.float32, device=device)
    x_va = np.concatenate(xg_va, axis=0)
    x_va = np.concatenate([x_va, np.concatenate(xa_va, axis=0)], axis=-1)
    x_va = torch.tensor(x_va, dtype=torch.float32, device=device)
    ylen_va = torch.tensor(np.concatenate(yl_va, axis=0), dtype=torch.float32, device=device)
    yhead_va = torch.tensor(np.concatenate(yh_va, axis=0), dtype=torch.float32, device=device)
    return x_tr, ylen_tr, yhead_tr, x_va, ylen_va, yhead_va


def load_data_2d_ronin(ronin_root, device, window_size=160, stride=32):
    """
    加载 RONIN 数据集并分割为训练集和验证集
    
    Args:
        ronin_root: RONIN 数据集根目录
        device: torch 设备
        window_size: 窗口大小
        stride: 步长
    
    Returns:
        x_tr, ylen_tr, yhead_tr, x_va, ylen_va, yhead_va
    """
    train_dirs = []
    for subset in ['train_dataset_1', 'train_dataset_2']:
        base = os.path.join(ronin_root, 'Data', subset)
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                d = os.path.join(base, name)
                if os.path.isdir(d):
                    train_dirs.append(d)
    val_dirs = []
    seen_base = os.path.join(ronin_root, 'Data', 'seen_subjects_test_set')
    if os.path.isdir(seen_base):
        for name in sorted(os.listdir(seen_base)):
            d = os.path.join(seen_base, name)
            if os.path.isdir(d):
                val_dirs.append(d)

    xg_tr, xa_tr, yl_tr, yh_tr = [], [], [], []
    xg_va, xa_va, yl_va, yh_va = [], [], [], []
    for d in train_dirs:
        gyro, acc, pos3d, ori = load_ronin_raw(d)
        
        [gx, ax], [dl, dh], _, _ = ronin_window(
            gyro, acc, pos3d, ori, mode='2d', window_size=window_size, stride=stride, filter_window=20,
            smooth_heading=True,  # 启用航向角平滑，提高真值轨迹光滑性
            heading_sigma=1.5,    # 航向角高斯平滑标准差
            smooth_length=False,  # 不平滑步长，只平滑航向
            length_sigma=1.5,    # 步长高斯平滑标准差
        )
        if gx.shape[0] == 0:
            continue
        xg_tr.append(gx); xa_tr.append(ax); yl_tr.append(dl); yh_tr.append(dh)
    for d in val_dirs:
        gyro, acc, pos3d, ori = load_ronin_raw(d)
        
        [gx, ax], [dl, dh], _, _ = ronin_window(
            gyro, acc, pos3d, ori, mode='2d', window_size=window_size, stride=stride, filter_window=20,
            smooth_heading=True,  # 启用航向角平滑，提高真值轨迹光滑性
            heading_sigma=1.25,    # 航向角高斯平滑标准差
            smooth_length=False,  # 不平滑步长，只平滑航向
            length_sigma=1.5,    # 步长高斯平滑标准差
        )
        if gx.shape[0] == 0:
            continue
        xg_va.append(gx); xa_va.append(ax); yl_va.append(dl); yh_va.append(dh)

    x_tr = np.concatenate(xg_tr, axis=0)
    x_tr = np.concatenate([x_tr, np.concatenate(xa_tr, axis=0)], axis=-1)
    x_tr = torch.tensor(x_tr, dtype=torch.float32, device=device)
    ylen_tr = torch.tensor(np.concatenate(yl_tr, axis=0), dtype=torch.float32, device=device)
    yhead_tr = torch.tensor(np.concatenate(yh_tr, axis=0), dtype=torch.float32, device=device)
    x_va = np.concatenate(xg_va, axis=0)
    x_va = np.concatenate([x_va, np.concatenate(xa_va, axis=0)], axis=-1)
    x_va = torch.tensor(x_va, dtype=torch.float32, device=device)
    ylen_va = torch.tensor(np.concatenate(yl_va, axis=0), dtype=torch.float32, device=device)
    yhead_va = torch.tensor(np.concatenate(yh_va, axis=0), dtype=torch.float32, device=device)
    return x_tr, ylen_tr, yhead_tr, x_va, ylen_va, yhead_va


def load_data_2d_ridi(ridi_root, device, window_size=160, stride=32):
    """
    加载 RIDI 数据集并分割为训练集和验证集

    Args:
        ridi_root: RIDI 数据集根目录
        device: torch 设备
        window_size: 窗口大小
        stride: 步长
        使用官方发布的 train/test 列表划分

    Returns:
        x_tr, ylen_tr, yhead_tr, x_va, ylen_va, yhead_va
    """
    data_root = os.path.join(ridi_root, "data")
    train_list = os.path.join(data_root, "list_train_publish_v2.txt")
    test_list = os.path.join(data_root, "list_test_publish_v2.txt")
    if not (os.path.exists(train_list) and os.path.exists(test_list)):
        raise RuntimeError("RIDI list_train/list_test files not found under RIDI/data")

    def _load_list(path):
        names = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                names.append(line.split(",")[0])
        return names

    train_names = _load_list(train_list)
    val_names = _load_list(test_list)

    xg_tr, xa_tr, yl_tr, yh_tr = [], [], [], []
    xg_va, xa_va, yl_va, yh_va = [], [], [], []

    for name in train_names + val_names:
        seq_dir = os.path.join(data_root, name)
        if not os.path.isdir(seq_dir):
            continue
        gyro, acc, pos3d, ori = load_ridi_raw(seq_dir)
        [gx, ax], [dl, dh], _, _ = ridi_window(
            gyro, acc, pos3d, ori,
            mode="2d",
            window_size=window_size,
            stride=stride,
            filter_window=20,
            smooth_heading=False,
            heading_sigma=1.5,
            smooth_length=False,
            length_sigma=1.0,
        )
        if name in val_names:
            xg_va.append(gx)
            xa_va.append(ax)
            yl_va.append(dl)
            yh_va.append(dh)
        else:
            xg_tr.append(gx)
            xa_tr.append(ax)
            yl_tr.append(dl)
            yh_tr.append(dh)

    x_tr = np.concatenate(xg_tr, axis=0)
    x_tr = np.concatenate([x_tr, np.concatenate(xa_tr, axis=0)], axis=-1)
    x_tr = torch.tensor(x_tr, dtype=torch.float32, device=device)
    ylen_tr = torch.tensor(np.concatenate(yl_tr, axis=0), dtype=torch.float32, device=device)
    yhead_tr = torch.tensor(np.concatenate(yh_tr, axis=0), dtype=torch.float32, device=device)

    x_va = np.concatenate(xg_va, axis=0)
    x_va = np.concatenate([x_va, np.concatenate(xa_va, axis=0)], axis=-1)
    x_va = torch.tensor(x_va, dtype=torch.float32, device=device)
    ylen_va = torch.tensor(np.concatenate(yl_va, axis=0), dtype=torch.float32, device=device)
    yhead_va = torch.tensor(np.concatenate(yh_va, axis=0), dtype=torch.float32, device=device)

    return x_tr, ylen_tr, yhead_tr, x_va, ylen_va, yhead_va


def load_data_2d_ridi_absheading(ridi_root, device, window_size=160, stride=32, return_ori=False, return_rel_ori=False, return_seq=False, return_init=False, return_delta_p=False, use_abs_heading=True, smooth_heading=False, heading_sigma=1.5, align_init_quat=False, acc_source="acce", align_init_quat_to_labels=True, return_align=False):
    """
    加载 RIDI 数据集并分割为训练/验证，返回绝对航向标签
    """
    data_root = os.path.join(ridi_root, "data")
    train_list = os.path.join(data_root, "list_train_publish_v2.txt")
    test_list = os.path.join(data_root, "list_test_publish_v2.txt")
    if not (os.path.exists(train_list) and os.path.exists(test_list)):
        raise RuntimeError("RIDI list_train/list_test files not found under RIDI/data")

    def _load_list(path):
        names = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                names.append(line.split(",")[0])
        return names

    train_names = _load_list(train_list)
    val_names = _load_list(test_list)

    xg_tr, xa_tr, yl_tr, ya_tr, yp_tr, yo_tr, yr_tr, yi_tr, yalign_tr = [], [], [], [], [], [], [], [], []
    xg_va, xa_va, yl_va, ya_va, yp_va, yo_va, yr_va, yi_va, yalign_va = [], [], [], [], [], [], [], [], []
    seq_tr = []
    seq_va = []

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
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.stack([w, x, y, z], axis=1)

    def _quat_to_rotmat(q):
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

    for name in train_names + val_names:
        seq_dir = os.path.join(data_root, name)
        if not os.path.isdir(seq_dir):
            continue
        gyro, acc, pos3d, ori = load_ridi_raw(seq_dir, acc_source=acc_source)
        q_align = None
        R_align = None
        if align_init_quat:
            q0 = ori[0].astype(np.float32)
            q_align = _quat_conj_batch(q0)[0]
            R_align = _quat_to_rotmat(q_align)
        # compute init rotation aligned to chord frame using sequence-level yaw offset
        pos2d = pos3d[:, :2]
        max_start = gyro.shape[0] - window_size - 1
        b_indices = []
        a_indices = []
        for idx in range(0, max_start, stride):
            a = idx + window_size // 2 - stride // 2
            b = idx + window_size // 2 + stride // 2
            a = max(0, min(a, len(pos2d) - 1))
            b = max(0, min(b, len(pos2d) - 1))
            a_indices.append(a)
            b_indices.append(b)
        a_indices = np.array(a_indices, dtype=np.int64)
        b_indices = np.array(b_indices, dtype=np.int64)

        if return_rel_ori and return_ori and return_delta_p:
            [gx, ax], [dl, y_head, abs_h, dori, drel, dp], _, init_head = ridi_window(
                gyro, acc, pos3d, ori,
                mode="2d",
                window_size=window_size,
                stride=stride,
                filter_window=0,
                smooth_heading=smooth_heading,
                heading_sigma=heading_sigma,
                smooth_length=False,
                length_sigma=1.0,
                return_abs_heading=True,
                return_ori=True,
                return_rel_ori=True,
                return_delta_p=True,
                abs_heading_from_ori=False,
                align_heading_to_init_pose=True,
            )
        elif return_rel_ori and return_delta_p:
            [gx, ax], [dl, y_head, abs_h, drel, dp], _, init_head = ridi_window(
                gyro, acc, pos3d, ori,
                mode="2d",
                window_size=window_size,
                stride=stride,
                filter_window=0,
                smooth_heading=smooth_heading,
                heading_sigma=heading_sigma,
                smooth_length=False,
                length_sigma=1.0,
                return_abs_heading=True,
                return_rel_ori=True,
                return_delta_p=True,
                abs_heading_from_ori=False,
                align_heading_to_init_pose=True,
            )
        elif return_rel_ori and return_ori:
            [gx, ax], [dl, y_head, abs_h, dori, drel], _, init_head = ridi_window(
                gyro, acc, pos3d, ori,
                mode="2d",
                window_size=window_size,
                stride=stride,
                filter_window=0,
                smooth_heading=smooth_heading,
                heading_sigma=heading_sigma,
                smooth_length=False,
                length_sigma=1.0,
                return_abs_heading=True,
                return_ori=True,
                return_rel_ori=True,
                abs_heading_from_ori=False,
                align_heading_to_init_pose=True,
            )
        elif return_rel_ori:
            [gx, ax], [dl, y_head, abs_h, drel], _, init_head = ridi_window(
                gyro, acc, pos3d, ori,
                mode="2d",
                window_size=window_size,
                stride=stride,
                filter_window=0,
                smooth_heading=smooth_heading,
                heading_sigma=heading_sigma,
                smooth_length=False,
                length_sigma=1.0,
                return_abs_heading=True,
                return_rel_ori=True,
                abs_heading_from_ori=False,
                align_heading_to_init_pose=True,
            )
        elif return_delta_p:
            [gx, ax], [dl, y_head, abs_h, dp], _, init_head = ridi_window(
                gyro, acc, pos3d, ori,
                mode="2d",
                window_size=window_size,
                stride=stride,
                filter_window=0,
                smooth_heading=smooth_heading,
                heading_sigma=heading_sigma,
                smooth_length=False,
                length_sigma=1.0,
                return_abs_heading=True,
                return_delta_p=True,
                abs_heading_from_ori=False,
                align_heading_to_init_pose=True,
            )
        else:
            [gx, ax], [dl, y_head, abs_h], _, init_head = ridi_window(
            gyro, acc, pos3d, ori,
            mode="2d",
            window_size=window_size,
            stride=stride,
            filter_window=0,
            smooth_heading=smooth_heading,
            heading_sigma=heading_sigma,
            smooth_length=False,
            length_sigma=1.0,
            return_abs_heading=True,
            abs_heading_from_ori=False,
            align_heading_to_init_pose=True,
            )

        if align_init_quat_to_labels and align_init_quat and dp is not None and dp.shape[1] == 3:
            dp = (R_align @ dp.T).T
        if align_init_quat_to_labels and align_init_quat and return_ori:
            dori = _quat_mul_batch(q_align, dori)

        init_q = ori[b_indices[0]].astype(np.float32)
        # quat to rot
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
        if align_init_quat_to_labels and align_init_quat and return_init:
            init_rot = R_align @ init_rot
        heading_label = abs_h if use_abs_heading else y_head
        if name in val_names:
            xg_va.append(gx)
            xa_va.append(ax)
            yl_va.append(dl)
            ya_va.append(heading_label)
            if return_delta_p:
                yp_va.append(dp)
            if return_ori:
                yo_va.append(dori)
            if return_rel_ori:
                yr_va.append(drel)
            if return_init:
                yi_va.append(np.repeat(init_rot[None, :, :], gx.shape[0], axis=0))
            if return_align and align_init_quat:
                yalign_va.append(np.repeat(R_align[None, :, :], gx.shape[0], axis=0))
            if return_seq:
                seq_va.append(np.full((gx.shape[0],), len(seq_va), dtype=np.int64))
        else:
            xg_tr.append(gx)
            xa_tr.append(ax)
            yl_tr.append(dl)
            ya_tr.append(heading_label)
            if return_delta_p:
                yp_tr.append(dp)
            if return_ori:
                yo_tr.append(dori)
            if return_rel_ori:
                yr_tr.append(drel)
            if return_init:
                yi_tr.append(np.repeat(init_rot[None, :, :], gx.shape[0], axis=0))
            if return_align and align_init_quat:
                yalign_tr.append(np.repeat(R_align[None, :, :], gx.shape[0], axis=0))
            if return_seq:
                seq_tr.append(np.full((gx.shape[0],), len(seq_tr), dtype=np.int64))

    x_tr = np.concatenate(xg_tr, axis=0)
    x_tr = np.concatenate([x_tr, np.concatenate(xa_tr, axis=0)], axis=-1)
    x_tr = torch.tensor(x_tr, dtype=torch.float32, device=device)
    ylen_tr = torch.tensor(np.concatenate(yl_tr, axis=0), dtype=torch.float32, device=device)
    yabs_tr = torch.tensor(np.concatenate(ya_tr, axis=0), dtype=torch.float32, device=device)
    ydp_tr = None
    if return_delta_p:
        ydp_tr = torch.tensor(np.concatenate(yp_tr, axis=0), dtype=torch.float32, device=device)
    yori_tr = None
    if return_ori:
        yori_tr = torch.tensor(np.concatenate(yo_tr, axis=0), dtype=torch.float32, device=device)
    yrel_tr = None
    if return_rel_ori:
        yrel_tr = torch.tensor(np.concatenate(yr_tr, axis=0), dtype=torch.float32, device=device)
    seqid_tr = None
    if return_seq:
        seqid_tr = torch.tensor(np.concatenate(seq_tr, axis=0), dtype=torch.int64, device=device)
    yinit_tr = None
    if return_init:
        yinit_tr = torch.tensor(np.concatenate(yi_tr, axis=0), dtype=torch.float32, device=device)
    yalign_tr_t = None
    if return_align and align_init_quat:
        yalign_tr_t = torch.tensor(np.concatenate(yalign_tr, axis=0), dtype=torch.float32, device=device)

    x_va = np.concatenate(xg_va, axis=0)
    x_va = np.concatenate([x_va, np.concatenate(xa_va, axis=0)], axis=-1)
    x_va = torch.tensor(x_va, dtype=torch.float32, device=device)
    ylen_va = torch.tensor(np.concatenate(yl_va, axis=0), dtype=torch.float32, device=device)
    yabs_va = torch.tensor(np.concatenate(ya_va, axis=0), dtype=torch.float32, device=device)
    ydp_va = None
    if return_delta_p:
        ydp_va = torch.tensor(np.concatenate(yp_va, axis=0), dtype=torch.float32, device=device)
    yori_va = None
    if return_ori:
        yori_va = torch.tensor(np.concatenate(yo_va, axis=0), dtype=torch.float32, device=device)
    yrel_va = None
    if return_rel_ori:
        yrel_va = torch.tensor(np.concatenate(yr_va, axis=0), dtype=torch.float32, device=device)
    seqid_va = None
    if return_seq:
        seqid_va = torch.tensor(np.concatenate(seq_va, axis=0), dtype=torch.int64, device=device)
    yinit_va = None
    if return_init:
        yinit_va = torch.tensor(np.concatenate(yi_va, axis=0), dtype=torch.float32, device=device)
    yalign_va_t = None
    if return_align and align_init_quat:
        yalign_va_t = torch.tensor(np.concatenate(yalign_va, axis=0), dtype=torch.float32, device=device)

    outputs = [x_tr, ylen_tr, yabs_tr]
    if return_delta_p:
        outputs.append(ydp_tr)
    if return_ori:
        outputs.append(yori_tr)
    if return_rel_ori:
        outputs.append(yrel_tr)
    if return_seq:
        outputs.append(seqid_tr)
    if return_init:
        outputs.append(yinit_tr)
    if return_align and align_init_quat:
        outputs.append(yalign_tr_t)
    outputs += [x_va, ylen_va, yabs_va]
    if return_delta_p:
        outputs.append(ydp_va)
    if return_ori:
        outputs.append(yori_va)
    if return_rel_ori:
        outputs.append(yrel_va)
    if return_seq:
        outputs.append(seqid_va)
    if return_init:
        outputs.append(yinit_va)
    if return_align and align_init_quat:
        outputs.append(yalign_va_t)
    return tuple(outputs)




def plot_quantizer_analysis(quantizer, heading_data, curve_dir, num_bins):
    """
    绘制量化器分析图
    
    Args:
        quantizer: 量化器对象
        heading_data: 航向角数据
        curve_dir: 输出目录
        num_bins: bin 数量
    """
    if not quantizer.fitted:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 航向角分布直方图
    heading_deg = np.degrees(heading_data)
    axes[0, 0].hist(heading_deg, bins=100, alpha=0.7, density=True, label='Data Distribution')
    axes[0, 0].set_title('Heading Angle Distribution')
    axes[0, 0].set_xlabel('Heading Change (deg)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    
    # 2. Bin 边界可视化
    bin_edges_deg = np.degrees(quantizer.bin_edges)
    bin_centers_deg = np.degrees(quantizer.bin_centers)
    
    axes[0, 1].scatter(range(len(bin_centers_deg)), bin_centers_deg, s=10, alpha=0.7)
    axes[0, 1].set_title('Bin Centers')
    axes[0, 1].set_xlabel('Bin Index')
    axes[0, 1].set_ylabel('Angle (deg)')
    
    # 3. Bin 宽度分布
    bin_widths = np.diff(quantizer.bin_edges)
    bin_widths_deg = np.degrees(bin_widths)
    
    axes[1, 0].bar(range(len(bin_widths_deg)), bin_widths_deg, alpha=0.7)
    uniform_width = np.degrees(2 * np.pi / num_bins)
    axes[1, 0].axhline(y=uniform_width, color='r', linestyle='--', 
                       label=f'Uniform: {uniform_width:.2f}deg')
    axes[1, 0].set_title('Bin Width Distribution')
    axes[1, 0].set_xlabel('Bin Index')
    axes[1, 0].set_ylabel('Width (deg)')
    axes[1, 0].legend()
    
    # 4. 小角度区域精度对比
    center_mask = np.abs(bin_centers_deg) < 30
    if center_mask.any():
        # bin_widths_deg 和 bin_centers_deg 长度相同
        center_widths = bin_widths_deg[center_mask]
        edge_widths = bin_widths_deg[~center_mask]
        
        if len(center_widths) > 0 and len(edge_widths) > 0:
            data = [center_widths, edge_widths]
            labels = ['Center (|angle|<30deg)', 'Edge']
            bp = axes[1, 1].boxplot(data, labels=labels)
            axes[1, 1].axhline(y=uniform_width, color='r', linestyle='--', label=f'Uniform: {uniform_width:.2f}deg')
            axes[1, 1].set_title('Bin Width: Center vs Edge')
            axes[1, 1].set_ylabel('Width (deg)')
            axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(curve_dir, 'quantizer_analysis.png'))
    plt.close()
    
    print(f"[Quantizer] Analysis saved to {curve_dir}/quantizer_analysis.png")
