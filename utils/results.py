"""
结果处理和评估工具函数
包含用于测试结果处理、分析和保存的函数
"""

import os
import numpy as np
import pandas as pd


def compute_path_length(traj: np.ndarray) -> float:
    """计算轨迹的总路径长度"""
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def reconstruct_from_absolute_angles(init_pos, step_lengths, absolute_angles):
    """
    根据步长和绝对角度重建轨迹
    
    Args:
        init_pos: 初始位置 (2,)
        step_lengths: 步长序列
        absolute_angles: 绝对角度序列
    
    Returns:
        traj: 重建的轨迹 (N, 2)
    """
    traj = [init_pos]
    curr_pos = init_pos.copy()
    # 确保长度一致
    n = min(len(step_lengths), len(absolute_angles))
    for i in range(n):
        l = step_lengths[i]
        theta = absolute_angles[i]
        dx = l * np.cos(theta)
        dy = l * np.sin(theta)
        curr_pos[0] += dx
        curr_pos[1] += dy
        traj.append(curr_pos.copy())
    return np.array(traj)




def extract_ground_truth_positions(pos_xyz, window_size, stride, num_windows, init_l):
    """
    从真值位置数据中提取与窗口对应的位置
    
    根据 window_dataset 的逻辑：
    - 初始位置在 mid = window_size // 2 - stride // 2
    - 窗口起始索引: idx = i * stride
    - 步长和航向角变化量是从 a 到 b 计算的：
      - a = idx + window_size // 2 - stride // 2
      - b = idx + window_size // 2 + stride // 2
    - 我们使用 b 位置作为该窗口的代表位置
    
    Args:
        pos_xyz: 真值位置数据 (N, 3)
        window_size: 窗口大小
        stride: 步长
        num_windows: 窗口数量
        init_l: 初始位置 (2,)
    
    Returns:
        gt_positions: 每个窗口对应的真值位置 (num_windows, 2)
    """
    pos_xy = pos_xyz[:, :2]
    gt_positions = [init_l]  # 第一个位置是初始位置
    
    for i in range(num_windows - 1):  # 从第二个窗口开始
        idx = i * stride
        # 计算窗口对应的位置索引 b
        b = idx + window_size // 2 + stride // 2
        # 确保索引在有效范围内
        b = min(max(0, b), len(pos_xy) - 1)
        gt_positions.append(pos_xy[b])
    
    return np.array(gt_positions)


def save_results_to_csv(gt_vis, pred_vis, traj_pdr, dl, dh, pred_len, pred_head,
                       vis_num, base_name, output_dir):
    """
    保存测试结果到CSV文件
    
    Args:
        gt_vis: 真值轨迹可视化数据
        pred_vis: 预测轨迹可视化数据
        traj_pdr: PDR轨迹
        dl: 真值步长变化量
        dh: 真值航向角变化量
        pred_len: 预测步长
        pred_head: 预测航向角
        vis_num: 可视化数量
        base_name: 基础文件名
        output_dir: 输出目录
    """
    df_traj = pd.DataFrame({
        "step": np.arange(len(gt_vis)),
        "gt_x": gt_vis[:, 0],
        "gt_y": gt_vis[:, 1],
        "pred_x": pred_vis[:, 0],
        "pred_y": pred_vis[:, 1],
    })
    df_traj.to_csv(os.path.join(output_dir, f"{base_name}_trajectory.csv"), index=False)

    df_pdr_traj = pd.DataFrame({
        "step": np.arange(len(traj_pdr)),
        "pdr_x": traj_pdr[:, 0],
        "pdr_y": traj_pdr[:, 1],
    })
    df_pdr_traj.to_csv(os.path.join(output_dir, f"{base_name}_pdr_trajectory.csv"), index=False)

    df_time = pd.DataFrame({
        "step": np.arange(vis_num),
        "gt_dl": dl[:vis_num, 0],
        "pred_dl": pred_len[:vis_num, 0],
        "gt_dh": dh[:vis_num, 0],
        "pred_dh": pred_head[:vis_num, 0],
    })
    df_time.to_csv(os.path.join(output_dir, f"{base_name}_time_series.csv"), index=False)
