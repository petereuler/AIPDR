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

