"""
可视化工具函数
包含所有用于测试结果可视化的函数
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def wrap_angle(angle):
    """将角度包装到 [-pi, pi] 范围"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ==================== 轨迹可视化 ====================

def plot_trajectory_comparison(traj_gt, traj_gt_xy, traj_pred, output_dir, base_name, traj_gt_raw=None, traj_pdr=None, vis_num=None, traj_pred_corrected=None):
    """
    绘制轨迹对比图 - 四个子图展示不同轨迹对比
    
    Args:
        traj_gt: 基于步长+航向角累积生成的平滑后真值轨迹
        traj_gt_xy: 基于真值(x,y)坐标提取的轨迹
        traj_pred: 预测轨迹（矫正前）
        output_dir: 输出目录
        base_name: 文件名前缀
        traj_gt_raw: 平滑前的真值轨迹（可选）
        traj_pdr: PDR轨迹（可选）
        vis_num: 可视化轨迹长度（可选，None表示使用完整长度）
        traj_pred_corrected: 矫正后的预测轨迹（可选，用于显示补偿效果）
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)
    
    # 如果没有指定vis_num，则使用最小长度
    if vis_num is None:
        vis_num = min(len(traj_gt), len(traj_gt_xy), len(traj_pred))
        if traj_gt_raw is not None:
            vis_num = min(vis_num, len(traj_gt_raw))
        if traj_pdr is not None:
            vis_num = min(vis_num, len(traj_pdr))

    # 1. 整体效果对比：traj_gt_xy, traj_gt_raw, traj_gt, traj_pred, traj_pdr
    ax1 = axes[0, 0]
    ax1.plot(traj_gt_xy[:vis_num, 0], traj_gt_xy[:vis_num, 1],
             'k-', linewidth=2, alpha=0.8, label='traj_gt_xy')
    if traj_gt_raw is not None:
        ax1.plot(traj_gt_raw[:vis_num, 0], traj_gt_raw[:vis_num, 1],
                 'g--', linewidth=1.5, alpha=0.7, label='traj_gt_raw')
    ax1.plot(traj_gt[:vis_num, 0], traj_gt[:vis_num, 1],
             'b-', linewidth=2, alpha=0.8, label='traj_gt')
    ax1.plot(traj_pred[:vis_num, 0], traj_pred[:vis_num, 1],
             'r--', linewidth=1.5, alpha=0.7, label='traj_pred (before correction)')
    if traj_pred_corrected is not None:
        ax1.plot(traj_pred_corrected[:vis_num, 0], traj_pred_corrected[:vis_num, 1],
                 'r-', linewidth=2, alpha=0.8, label='traj_pred (after correction)')
    if traj_pdr is not None:
        ax1.plot(traj_pdr[:vis_num, 0], traj_pdr[:vis_num, 1],
                 'orange', linewidth=1.5, alpha=0.6, label='traj_pdr')

    ax1.scatter(traj_gt_xy[0, 0], traj_gt_xy[0, 1], c='green', s=80, marker='o',
                edgecolors='white', zorder=6, label='Start')
    ax1.scatter(traj_gt_xy[-1, 0], traj_gt_xy[-1, 1], c='purple', s=80, marker='X',
                edgecolors='white', zorder=6, label='End')

    title_parts = ['traj_gt_xy', 'traj_gt_raw', 'traj_gt', 'traj_pred']
    if traj_pred_corrected is not None:
        title_parts.append('traj_pred_corrected')
    if traj_pdr is not None:
        title_parts.append('traj_pdr')
    ax1.set_title(f'Overall Comparison: {", ".join(title_parts)}', fontsize=12)
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.axis('equal')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. PDR逻辑与XY坐标提取对比：traj_gt_xy vs traj_gt_raw
    ax2 = axes[0, 1]
    ax2.plot(traj_gt_xy[:vis_num, 0], traj_gt_xy[:vis_num, 1],
             'k-', linewidth=2, alpha=0.8, label='traj_gt_xy (XY extraction)')
    if traj_gt_raw is not None:
        ax2.plot(traj_gt_raw[:vis_num, 0], traj_gt_raw[:vis_num, 1],
                 'g--', linewidth=2, alpha=0.8, label='traj_gt_raw (PDR logic)')

        # 计算差异
        diff_xy_raw = np.linalg.norm(traj_gt_xy[:vis_num] - traj_gt_raw[:vis_num], axis=1)
        mean_diff = np.mean(diff_xy_raw)
        ax2.set_title(f'XY vs PDR Logic\nMean diff: {mean_diff:.3f}m', fontsize=12)
    else:
        ax2.set_title('XY vs PDR Logic\n(No traj_gt_raw available)', fontsize=12)

    ax2.scatter(traj_gt_xy[0, 0], traj_gt_xy[0, 1], c='green', s=60, marker='o',
                edgecolors='white', zorder=6)
    ax2.scatter(traj_gt_xy[-1, 0], traj_gt_xy[-1, 1], c='purple', s=60, marker='X',
                edgecolors='white', zorder=6)

    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.axis('equal')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # 3. 平滑前后对比：traj_gt_raw vs traj_gt
    ax3 = axes[1, 0]
    if traj_gt_raw is not None:
        ax3.plot(traj_gt_raw[:vis_num, 0], traj_gt_raw[:vis_num, 1],
                 'g--', linewidth=2, alpha=0.8, label='traj_gt_raw (Before smoothing)')
    ax3.plot(traj_gt[:vis_num, 0], traj_gt[:vis_num, 1],
             'b-', linewidth=2, alpha=0.8, label='traj_gt (After smoothing)')

    if traj_gt_raw is not None:
        # 计算平滑前后的差异
        diff_raw_smooth = np.linalg.norm(traj_gt_raw[:vis_num] - traj_gt[:vis_num], axis=1)
        mean_diff_smooth = np.mean(diff_raw_smooth)
        ax3.set_title(f'Before vs After Smoothing\nMean diff: {mean_diff_smooth:.3f}m', fontsize=12)
    else:
        ax3.set_title('Before vs After Smoothing\n(No traj_gt_raw available)', fontsize=12)
    
    ax3.scatter(traj_gt[0, 0], traj_gt[0, 1], c='green', s=60, marker='o',
                edgecolors='white', zorder=6)
    ax3.scatter(traj_gt[-1, 0], traj_gt[-1, 1], c='purple', s=60, marker='X',
                edgecolors='white', zorder=6)

    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    ax3.axis('equal')
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # 4. 模型学习效果：traj_gt vs traj_pred (矫正前后对比)
    ax4 = axes[1, 1]
    ax4.plot(traj_gt[:vis_num, 0], traj_gt[:vis_num, 1],
             'b-', linewidth=2, alpha=0.8, label='traj_gt (Ground Truth)')

    if traj_pred_corrected is not None:
        # 显示矫正前后的对比
        ax4.plot(traj_pred[:vis_num, 0], traj_pred[:vis_num, 1],
                 'r--', linewidth=1.5, alpha=0.7, label='traj_pred (before correction)')
        ax4.plot(traj_pred_corrected[:vis_num, 0], traj_pred_corrected[:vis_num, 1],
                 'r-', linewidth=2, alpha=0.8, label='traj_pred (after correction)')

        # 计算两种预测的误差
        pred_errors_before = np.linalg.norm(traj_gt[:vis_num] - traj_pred[:vis_num], axis=1)
        pred_errors_after = np.linalg.norm(traj_gt[:vis_num] - traj_pred_corrected[:vis_num], axis=1)
        rmse_before = np.sqrt(np.mean(pred_errors_before ** 2))
        rmse_after = np.sqrt(np.mean(pred_errors_after ** 2))
        improvement = (rmse_before - rmse_after) / rmse_before * 100

        ax4.set_title(f'Model Learning: GT vs Prediction\nBefore: RMSE {rmse_before:.3f}m, After: {rmse_after:.3f}m\nImprovement: {improvement:.1f}%', fontsize=10)
    else:
        # 只有一种预测的情况（向后兼容）
        ax4.plot(traj_pred[:vis_num, 0], traj_pred[:vis_num, 1],
                 'r-', linewidth=2, alpha=0.8, label='traj_pred (Model Prediction)')

        pred_errors = np.linalg.norm(traj_gt[:vis_num] - traj_pred[:vis_num], axis=1)
        rmse = np.sqrt(np.mean(pred_errors ** 2))
        mean_error = np.mean(pred_errors)
    
        ax4.set_title(f'Model Learning: GT vs Prediction\nRMSE: {rmse:.3f}m, Mean: {mean_error:.3f}m', fontsize=12)

    ax4.scatter(traj_gt[0, 0], traj_gt[0, 1], c='green', s=60, marker='o',
                edgecolors='white', zorder=6)
    ax4.scatter(traj_gt[-1, 0], traj_gt[-1, 1], c='purple', s=60, marker='X',
                edgecolors='white', zorder=6)

    ax4.set_xlabel('X (m)', fontsize=10)
    ax4.set_ylabel('Y (m)', fontsize=10)
    ax4.axis('equal')
    ax4.legend(fontsize=8, loc='best')
    ax4.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle(f'Trajectory Comparison Analysis - {base_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_trajectory.png"), bbox_inches='tight')
    plt.close()
    
    print(f"  轨迹对比图保存至: {base_name}_trajectory.png")



# ==================== 航向角可视化 ====================

def plot_heading_analysis(dh, pred_head_soft, pred_head_hard, vis_num, output_path, quantizer, dh_raw=None):
    """绘制航向分类分析图
    
    Args:
        dh: 平滑后的航向角真值
        pred_head_soft: 预测航向角（软解码）
        pred_head_hard: 预测航向角（硬解码）
        vis_num: 可视化数量
        output_path: 输出路径
        quantizer: 量化器
        dh_raw: 平滑前的航向角真值（可选）
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=150)
    t = np.arange(vis_num)
    
    # 1. 航向时间序列对比
    ax1 = axes[0]
    if dh_raw is not None:
        min_len = min(len(dh_raw), vis_num)
        ax1.plot(t[:min_len], np.degrees(dh_raw[:min_len, 0]), label='Ground Truth (Raw)', 
                linewidth=1.2, color='gray', linestyle='--', alpha=0.6)
    ax1.plot(t, np.degrees(dh[:vis_num, 0]), label='Ground Truth (Smoothed)', linewidth=1.5, color='black')
    ax1.plot(t, np.degrees(pred_head_soft[:vis_num, 0]), label='Predicted (Soft)', 
             linewidth=1.2, color='red', alpha=0.8)
    ax1.set_title("Heading Change Over Time", fontsize=16)
    ax1.set_ylabel("Heading Change (deg)", fontsize=14)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(linestyle=':', alpha=0.6)
    
    # 2. 硬解码 vs 软解码
    ax2 = axes[1]
    if dh_raw is not None:
        min_len = min(len(dh_raw), vis_num)
        ax2.plot(t[:min_len], np.degrees(dh_raw[:min_len, 0]), label='Ground Truth (Raw)', 
                linewidth=1.2, color='gray', linestyle='--', alpha=0.6)
    ax2.plot(t, np.degrees(dh[:vis_num, 0]), label='Ground Truth (Smoothed)', linewidth=1.5, color='black')
    ax2.plot(t, np.degrees(pred_head_hard[:vis_num, 0]), label='Hard Decode (argmax)', 
             linewidth=1.2, linestyle='--', color='tab:orange')
    ax2.plot(t, np.degrees(pred_head_soft[:vis_num, 0]), label='Soft Decode', 
             linewidth=1.2, color='tab:green')
    ax2.set_title("Hard vs Soft Decode Comparison", fontsize=16)
    ax2.set_ylabel("Heading Change (deg)", fontsize=14)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(linestyle=':', alpha=0.6)
    
    # 3. 误差分析
    ax3 = axes[2]
    error_soft = np.abs(wrap_angle(pred_head_soft[:vis_num, 0] - dh[:vis_num, 0]))
    error_hard = np.abs(wrap_angle(pred_head_hard[:vis_num, 0] - dh[:vis_num, 0]))
    
    ax3.plot(t, np.degrees(error_soft), label=f'Soft Decode MAE: {np.degrees(error_soft.mean()):.2f}deg', 
             linewidth=1.0, color='red', alpha=0.8)
    ax3.plot(t, np.degrees(error_hard), label=f'Hard Decode MAE: {np.degrees(error_hard.mean()):.2f}deg', 
             linewidth=1.0, color='tab:orange', alpha=0.6)
    
    # 显示自适应量化信息
    if getattr(quantizer, "adaptive", False) and getattr(quantizer, "fitted", False):
        bin_widths = np.diff(quantizer.bin_edges)
        min_width = np.degrees(bin_widths.min())
        max_width = np.degrees(bin_widths.max())
        ax3.axhline(y=min_width/2, color='green', linestyle=':', 
                    label=f'Min Quant Error: +/-{min_width/2:.2f}deg')
        ax3.axhline(y=max_width/2, color='gray', linestyle=':', 
                    label=f'Max Quant Error: +/-{max_width/2:.2f}deg')
    
    ax3.set_title("Heading Prediction Error", fontsize=16)
    ax3.set_ylabel("Absolute Error (deg)", fontsize=14)
    ax3.set_xlabel("Window Index", fontsize=14)
    ax3.legend(fontsize=12, loc='upper right')
    ax3.grid(linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


# ==================== 时间序列可视化 ====================

def plot_time_series(dl, dh, pred_len, pred_head, vis_num, output_path, dl_raw=None, dh_raw=None):
    """绘制时间序列图
    
    Args:
        dl: 平滑后的步长真值
        dh: 平滑后的航向角真值
        pred_len: 预测步长
        pred_head: 预测航向角
        vis_num: 可视化数量
        output_path: 输出路径
        dl_raw: 平滑前的步长真值（可选）
        dh_raw: 平滑前的航向角真值（可选）
    """
    fig_t = plt.figure(figsize=(14, 15), dpi=150)
    ax1 = fig_t.add_subplot(3, 1, 1)
    ax2 = fig_t.add_subplot(3, 1, 2)
    ax3 = fig_t.add_subplot(3, 1, 3)
    t = np.arange(vis_num)
    
    # 步长时间序列
    if dl_raw is not None:
        min_len = min(len(dl_raw), vis_num)
        ax1.plot(t[:min_len], dl_raw[:min_len, 0], label='Ground Truth (Raw)', 
                linewidth=1.2, color='gray', linestyle='--', alpha=0.6)
    ax1.plot(t, dl[:vis_num, 0], label='Ground Truth (Smoothed)', linewidth=1.5, color='black')
    ax1.plot(t, pred_len[:vis_num, 0], label='Predicted', linewidth=1.2, color='red')
    ax1.set_title("Step Length Over Time", fontsize=20)
    ax1.set_ylabel("Step Length (m)", fontsize=16)
    ax1.legend(fontsize=14, loc='upper right')
    ax1.grid(linestyle=':', alpha=0.6)
    
    # 航向角时间序列
    if dh_raw is not None:
        min_len = min(len(dh_raw), vis_num)
        ax2.plot(t[:min_len], dh_raw[:min_len, 0], label='Ground Truth (Raw)', 
                linewidth=1.2, color='gray', linestyle='--', alpha=0.6)
    ax2.plot(t, dh[:vis_num, 0], label='Ground Truth (Smoothed)', linewidth=1.5, color='black')
    ax2.plot(t, pred_head[:vis_num, 0], label='Predicted', linewidth=1.2, color='red')
    ax2.set_title("Heading Change Over Time (Adaptive Quantization)", fontsize=20)
    ax2.set_ylabel("Heading Change (rad)", fontsize=16)
    ax2.legend(fontsize=14, loc='upper right')
    ax2.grid(linestyle=':', alpha=0.6)

    # 步长和航向角乘积时间序列
    # 计算真值乘积
    gt_product = dl[:vis_num, 0] * dh[:vis_num, 0]
    pred_product = pred_len[:vis_num, 0] * pred_head[:vis_num, 0]

    # 如果有原始数据，也计算原始乘积
    if dl_raw is not None and dh_raw is not None:
        min_len = min(len(dl_raw), len(dh_raw), vis_num)
        raw_product = dl_raw[:min_len, 0] * dh_raw[:min_len, 0]
        ax3.plot(t[:min_len], raw_product, label='Ground Truth (Raw)',
                linewidth=1.2, color='gray', linestyle='--', alpha=0.6)

    ax3.plot(t, gt_product, label='Ground Truth (Smoothed)', linewidth=1.5, color='black')
    ax3.plot(t, pred_product, label='Predicted', linewidth=1.2, color='red')
    ax3.set_title("Step Length × Heading Change Product", fontsize=20)
    ax3.set_ylabel("Product (m·rad)", fontsize=16)
    ax3.set_xlabel("Window Index", fontsize=16)
    ax3.legend(fontsize=14, loc='upper right')
    ax3.grid(linestyle=':', alpha=0.6)
    
    fig_t.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig_t)


def plot_cumulative_series(dl, dh, pred_len, pred_head, vis_num, init_h, output_path, dl_raw=None, dh_raw=None, init_h_raw=None):
    """绘制累积序列图
    
    Args:
        dl: 平滑后的步长真值
        dh: 平滑后的航向角真值
        pred_len: 预测步长
        pred_head: 预测航向角
        vis_num: 可视化数量
        init_h: 平滑后的初始航向角
        output_path: 输出路径
        dl_raw: 平滑前的步长真值（可选）
        dh_raw: 平滑前的航向角真值（可选）
        init_h_raw: 平滑前的初始航向角（可选）
    """
    fig_c = plt.figure(figsize=(14, 10), dpi=150)
    ax1 = fig_c.add_subplot(2, 1, 1)
    ax2 = fig_c.add_subplot(2, 1, 2)
    n = min(vis_num, len(dl), len(dh), len(pred_len), len(pred_head))
    t = np.arange(n)

    # 累积步长
    cum_gt_len = np.cumsum(dl[:n, 0])
    cum_pred_len = np.cumsum(pred_len[:n, 0])

    if dl_raw is not None:
        min_len = min(len(dl_raw), n)
        cum_gt_len_raw = np.cumsum(dl_raw[:min_len, 0])
        ax1.plot(t[:min_len], cum_gt_len_raw, label='Ground Truth (Raw)', 
                linewidth=1.2, color='gray', linestyle='--', alpha=0.6)

    ax1.plot(t, cum_gt_len, label='Ground Truth (Smoothed)', linewidth=1.4, color='black')
    ax1.plot(t, cum_pred_len, label='Predicted', linewidth=1.2, color='red')
    ax1.set_title('Cumulative Step Length', fontsize=24)
    ax1.set_ylabel('Length (m)', fontsize=20)
    ax1.legend(fontsize=16, loc='upper left')
    ax1.grid()
    ax1.tick_params(axis='both', labelsize=16)

    # 累积航向角
    gt_incr = dh[:n, 0]
    pred_incr = pred_head[:n, 0]
    cum_gt_head = init_h + np.unwrap(gt_incr).cumsum()
    cum_pred_head = init_h + np.unwrap(pred_incr).cumsum()
    
    if dh_raw is not None and init_h_raw is not None:
        min_len = min(len(dh_raw), n)
        gt_incr_raw = dh_raw[:min_len, 0]
        cum_gt_head_raw = init_h_raw + np.unwrap(gt_incr_raw).cumsum()
        ax2.plot(t[:min_len], np.degrees(cum_gt_head_raw), label='Ground Truth (Raw)', 
                linewidth=1.2, color='gray', linestyle='--', alpha=0.6)
    cum_gt_head = np.degrees(wrap_angle(cum_gt_head))
    cum_pred_head = np.degrees(wrap_angle(cum_pred_head))

    ax2.plot(t, cum_gt_head, label='Ground Truth', linewidth=1.4, color='black')
    ax2.plot(t, cum_pred_head, label='Predicted', linewidth=1.2, color='red')
    ax2.set_title('Cumulative Heading', fontsize=24)
    ax2.set_ylabel('Heading (deg)', fontsize=20)
    ax2.set_xlabel('Window Index', fontsize=20)
    ax2.legend(fontsize=16, loc='upper left')
    ax2.grid()
    ax2.tick_params(axis='both', labelsize=16)

    fig_c.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig_c)


def plot_cumulative_error_series(dl, dh, pred_len, pred_head, vis_num, init_h, output_path, dl_raw=None, dh_raw=None, init_h_raw=None):
    """绘制累积误差序列图（使用平滑前的真值计算误差）
    
    Args:
        dl: 平滑后的步长真值（用于显示，但不用于误差计算）
        dh: 平滑后的航向角真值（用于显示，但不用于误差计算）
        pred_len: 预测步长
        pred_head: 预测航向角
        vis_num: 可视化数量
        init_h: 平滑后的初始航向角
        output_path: 输出路径
        dl_raw: 平滑前的步长真值（用于误差计算）
        dh_raw: 平滑前的航向角真值（用于误差计算）
        init_h_raw: 平滑前的初始航向角（用于误差计算）
    """
    fig_e = plt.figure(figsize=(14, 10), dpi=150)
    ax1 = fig_e.add_subplot(2, 1, 1)
    ax2 = fig_e.add_subplot(2, 1, 2)
    
    # 使用平滑前的真值计算误差
    dl_gt = dl_raw if dl_raw is not None else dl
    dh_gt = dh_raw if dh_raw is not None else dh
    init_h_gt = init_h_raw if init_h_raw is not None else init_h
    
    n = min(vis_num, len(dl_gt), len(dh_gt), len(pred_len), len(pred_head))
    t = np.arange(n)

    e_len = np.abs(pred_len[:n, 0] - dl_gt[:n, 0])
    cum_e_len = np.cumsum(e_len)

    ax1.plot(t, cum_e_len, label='Cumulative Error', linewidth=1.4, color='tab:red')
    ax1.set_title('Cumulative Step Length Error (vs Raw GT)', fontsize=24)
    ax1.set_ylabel('Error (m)', fontsize=20)
    ax1.legend(fontsize=16, loc='upper left')
    ax1.grid()
    ax1.tick_params(axis='both', labelsize=16)

    pred_abs = init_h_gt + np.unwrap(pred_head[:n, 0]).cumsum()
    gt_abs = init_h_gt + np.unwrap(dh_gt[:n, 0]).cumsum()
    e_head = np.abs(wrap_angle(pred_abs - gt_abs))
    cum_e_head = np.cumsum(np.degrees(e_head))

    ax2.plot(t, cum_e_head, label='Cumulative Error', linewidth=1.4, color='tab:red')
    ax2.set_title('Cumulative Heading Error (vs Raw GT)', fontsize=24)
    ax2.set_ylabel('Error (deg)', fontsize=20)
    ax2.set_xlabel('Window Index', fontsize=20)
    ax2.legend(fontsize=16, loc='upper left')
    ax2.grid()
    ax2.tick_params(axis='both', labelsize=16)

    fig_e.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig_e)


def plot_error_histogram(dl, dh, pred_len, pred_head, vis_num, output_path, quantizer, dl_raw=None, dh_raw=None):
    """绘制误差直方图（使用平滑前的真值计算误差）
    
    Args:
        dl: 平滑后的步长真值（用于显示，但不用于误差计算）
        dh: 平滑后的航向角真值（用于显示，但不用于误差计算）
        pred_len: 预测步长
        pred_head: 预测航向角
        vis_num: 可视化数量
        output_path: 输出路径
        quantizer: 量化器
        dl_raw: 平滑前的步长真值（用于误差计算）
        dh_raw: 平滑前的航向角真值（用于误差计算）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 使用平滑前的真值计算误差
    dl_gt = dl_raw if dl_raw is not None else dl
    dh_gt = dh_raw if dh_raw is not None else dh
    
    # 步长误差
    len_error = pred_len[:vis_num, 0] - dl_gt[:vis_num, 0]
    axes[0].hist(len_error, bins=50, alpha=0.7, color='tab:blue', edgecolor='k')
    axes[0].axvline(len_error.mean(), color='red', linestyle='--', 
                   label=f'Mean: {len_error.mean():.4f}m')
    axes[0].axvline(0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_xlabel("Step Length Error (m)", fontsize=14)
    axes[0].set_ylabel("Frequency", fontsize=14)
    axes[0].set_title("Step Length Error Distribution (vs Raw GT)", fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 航向误差
    head_error = wrap_angle(pred_head[:vis_num, 0] - dh_gt[:vis_num, 0])
    head_error_deg = np.degrees(head_error)
    axes[1].hist(head_error_deg, bins=50, alpha=0.7, color='tab:green', edgecolor='k')
    axes[1].axvline(head_error_deg.mean(), color='red', linestyle='--', 
                   label=f'Mean: {head_error_deg.mean():.2f}deg')
    axes[1].axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # 量化误差参考线（自适应量化）- 仅在有量化器时显示
    if quantizer is not None and getattr(quantizer, "adaptive", False) and getattr(quantizer, "fitted", False):
        bin_widths = np.diff(quantizer.bin_edges)
        median_width = np.degrees(np.median(bin_widths))
        axes[1].axvline(median_width/2, color='gray', linestyle=':', 
                       label=f'Median Quant: +/-{median_width/2:.2f}deg')
        axes[1].axvline(-median_width/2, color='gray', linestyle=':')
    
    axes[1].set_xlabel("Heading Error (deg)", fontsize=14)
    axes[1].set_ylabel("Frequency", fontsize=14)
    axes[1].set_title("Heading Error Distribution (vs Raw GT)", fontsize=16)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ==================== 编码错误分析可视化 ====================

def analyze_encoding_errors(gt_heading, pred_binary_probs, pred_binary_hard, quantizer, output_bits, output_dir, file_prefix):
    """
    分析编码错误情况
    
    Args:
        gt_heading: 真值航向角 (N,)
        pred_binary_probs: 预测概率 (N, output_bits)
        pred_binary_hard: 预测硬判决 (N, output_bits)
        quantizer: 量化器
        output_bits: 输出位数
        output_dir: 输出目录
        file_prefix: 文件前缀
    """
    # 1. 计算真值编码
    if hasattr(quantizer, "angle_to_codeword"):
        gt_binary = quantizer.angle_to_codeword(
            torch.tensor(gt_heading, dtype=torch.float32)
        ).cpu().numpy()
    else:
        gt_binary = quantizer.encode_to_binary_vector(gt_heading)
    
    gt_binary_hard = (gt_binary > 0.5).astype(np.int32)
    
    # 2. 计算每个位的错误率
    bit_errors = (pred_binary_hard != gt_binary_hard).astype(np.int32)  # (N, output_bits)
    bit_error_rates = bit_errors.mean(axis=0)  # (output_bits,)
    
    # 3. 计算每个样本的错误位数
    num_bit_errors_per_sample = bit_errors.sum(axis=1)  # (N,)
    
    # 4. 统计完全正确的样本比例
    perfect_samples = (num_bit_errors_per_sample == 0).sum()
    perfect_rate = perfect_samples / len(gt_heading)
    
    # 5. 创建统计表格
    stats_data = {
        'Bit Index': list(range(output_bits)),
        'Error Rate': bit_error_rates.tolist(),
        'Total Errors': bit_errors.sum(axis=0).tolist(),
        'Total Samples': [len(gt_heading)] * output_bits
    }
    
    df_stats = pd.DataFrame(stats_data)
    
    # 6. 错误位数分布统计
    error_dist = {}
    for num_err in range(output_bits + 1):
        count = (num_bit_errors_per_sample == num_err).sum()
        error_dist[num_err] = count
    
    df_error_dist = pd.DataFrame({
        'Num Errors': list(error_dist.keys()),
        'Count': list(error_dist.values()),
        'Percentage': [f"{100 * v / len(gt_heading):.2f}%" for v in error_dist.values()]
    })
    
    # 7. 保存表格
    stats_csv = os.path.join(output_dir, f"{file_prefix}_encoding_stats.csv")
    df_stats.to_csv(stats_csv, index=False, float_format='%.6f')
    
    error_dist_csv = os.path.join(output_dir, f"{file_prefix}_error_distribution.csv")
    df_error_dist.to_csv(error_dist_csv, index=False)
    
    # 8. 可视化（合并所有图到一个大图）
    fig = plt.figure(figsize=(20, 14))
    
    # 8.1 每个位的错误率
    ax1 = plt.subplot(3, 3, 1)
    bars = ax1.bar(range(output_bits), bit_error_rates * 100, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Bit Index', fontsize=12)
    ax1.set_ylabel('Error Rate (%)', fontsize=12)
    ax1.set_title(f'Bit Error Rate (Avg: {bit_error_rates.mean()*100:.2f}%)', fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_xticks(range(0, output_bits, max(1, output_bits // 10)))
    
    # 添加数值标签
    for i, (bar, rate) in enumerate(zip(bars, bit_error_rates)):
        if rate > 0.05:  # 只显示错误率较高的
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 8.2 错误位数分布（改进版：添加百分比和说明）
    ax2 = plt.subplot(3, 3, 2)
    max_show_errors = min(11, output_bits + 1)
    error_counts = [error_dist.get(i, 0) for i in range(max_show_errors)]
    error_percentages = [100 * count / len(gt_heading) for count in error_counts]
    
    bars = ax2.bar(range(len(error_counts)), error_counts, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Number of Bit Errors per Sample', fontsize=12)
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_title(f'Error Distribution\n(Perfect: {perfect_rate*100:.1f}%, Avg: {num_bit_errors_per_sample.mean():.2f} errors/sample)', fontsize=14)
    ax2.set_xticks(range(len(error_counts)))
    ax2.set_xticklabels([str(i) for i in range(len(error_counts))])
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 在柱状图上添加百分比标签
    for i, (bar, pct) in enumerate(zip(bars, error_percentages)):
        if error_counts[i] > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 8.3 累积错误率（改进版：更清晰的说明）
    ax3 = plt.subplot(3, 3, 3)
    sorted_indices = np.argsort(bit_error_rates)[::-1]  # 按错误率降序排列的索引
    sorted_errors = bit_error_rates[sorted_indices]
    cumsum_errors = np.cumsum(sorted_errors)
    
    # 绘制累积错误率
    ax3.plot(range(output_bits), cumsum_errors * 100, 
             marker='o', markersize=4, linewidth=2, color='green', label='Cumulative Error Rate')
    ax3.axhline(y=bit_error_rates.mean() * 100, color='red', 
                linestyle='--', linewidth=1.5, label=f'Average: {bit_error_rates.mean()*100:.2f}%')
    
    # 添加说明文本
    ax3.text(0.02, 0.98, 
            f'Top 3 worst bits:\nBit {sorted_indices[0]}: {sorted_errors[0]*100:.1f}%\n'
            f'Bit {sorted_indices[1]}: {sorted_errors[1]*100:.1f}%\n'
            f'Bit {sorted_indices[2]}: {sorted_errors[2]*100:.1f}%',
            transform=ax3.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3.set_xlabel('Bit Index (Sorted by Error Rate, Worst → Best)', fontsize=12)
    ax3.set_ylabel('Cumulative Error Rate (%)', fontsize=12)
    ax3.set_title('Cumulative Error Rate\n(Shows total error rate if we fix worst N bits)', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(linestyle='--', alpha=0.5)
    
    # 8.4 概率分布热图（改进版：显示错误模式）
    ax4 = plt.subplot(3, 3, 4)
    n_show = min(100, len(pred_binary_probs))
    
    # 创建对比图：显示预测概率与真值的差异
    # 绿色=正确预测(高置信度), 红色=错误预测(高置信度), 黄色=低置信度
    comparison_probs = np.zeros((n_show, output_bits, 3))
    for i in range(n_show):
        for j in range(output_bits):
            gt_bit = gt_binary_hard[i, j]
            pred_prob = pred_binary_probs[i, j]
            pred_bit = pred_binary_hard[i, j]
            
            if gt_bit == pred_bit:
                # 正确预测：绿色，亮度表示置信度
                comparison_probs[i, j] = [0, pred_prob, 0]
            else:
                # 错误预测：红色，亮度表示置信度
                comparison_probs[i, j] = [pred_prob, 0, 0]
    
    ax4.imshow(comparison_probs, aspect='auto', interpolation='nearest')
    ax4.set_xlabel('Sample Index', fontsize=12)
    ax4.set_ylabel('Bit Index', fontsize=12)
    ax4.set_title(f'Prediction vs Ground Truth\n(Green=Correct, Red=Wrong, Brightness=Confidence)', fontsize=14)
    ax4.text(0.02, 0.98, f'First {n_show} samples\nEach row = one bit\nEach column = one sample',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 8.5 真值 vs 预测编码对比（前50个样本）
    ax5 = plt.subplot(3, 3, 5)
    n_show2 = min(50, len(gt_binary_hard))
    comparison = np.zeros((n_show2, output_bits, 3))
    for i in range(n_show2):
        for j in range(output_bits):
            if gt_binary_hard[i, j] == 1 and pred_binary_hard[i, j] == 1:
                comparison[i, j] = [0, 1, 0]  # 绿色：都正确
            elif gt_binary_hard[i, j] == 0 and pred_binary_hard[i, j] == 0:
                comparison[i, j] = [0, 1, 0]  # 绿色：都正确
            elif gt_binary_hard[i, j] == 1 and pred_binary_hard[i, j] == 0:
                comparison[i, j] = [1, 0, 0]  # 红色：应该是1但预测为0
            else:
                comparison[i, j] = [1, 1, 0]  # 黄色：应该是0但预测为1
    ax5.imshow(comparison, aspect='auto', interpolation='nearest')
    ax5.set_xlabel('Bit Index', fontsize=12)
    ax5.set_ylabel('Sample Index', fontsize=12)
    ax5.set_title(f'GT vs Pred Comparison (First {n_show2} samples)', fontsize=14)
    ax5.text(0.02, 0.98, 'Green: Correct\nRed: 1→0 error\nYellow: 0→1 error',
             transform=ax5.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 8.6 错误位数 vs 角度误差散点图（改进版：添加趋势线和统计）
    ax6 = plt.subplot(3, 3, 6)
    # 计算角度误差
    if hasattr(quantizer, "decode_soft_expectation"):
        logits = np.log(pred_binary_probs + 1e-8) - np.log(1.0 - pred_binary_probs + 1e-8)
        pred_heading = quantizer.decode_soft_expectation(
            torch.tensor(logits, dtype=torch.float32)
        ).cpu().numpy()
    else:
        pred_heading = quantizer.decode_from_binary_vector(pred_binary_probs)
    
    angle_errors = np.abs(wrap_angle(pred_heading - gt_heading))
    angle_errors_deg = np.degrees(angle_errors)
    
    # 计算每个错误位数的平均角度误差
    unique_error_counts = np.unique(num_bit_errors_per_sample)
    mean_angle_errors = []
    std_angle_errors = []
    for err_count in unique_error_counts:
        mask = num_bit_errors_per_sample == err_count
        mean_angle_errors.append(angle_errors_deg[mask].mean())
        std_angle_errors.append(angle_errors_deg[mask].std())
    
    # 散点图
    scatter = ax6.scatter(num_bit_errors_per_sample, angle_errors_deg, 
                         alpha=0.3, s=8, c=num_bit_errors_per_sample, 
                         cmap='viridis', edgecolors='none')
    
    # 添加趋势线（每个错误位数的平均值）
    ax6.plot(unique_error_counts, mean_angle_errors, 'ro-', 
            linewidth=2, markersize=8, label='Mean Angle Error', zorder=5)
    
    # 添加误差棒
    ax6.errorbar(unique_error_counts, mean_angle_errors, yerr=std_angle_errors,
                fmt='none', color='red', capsize=5, capthick=2, zorder=4)
    
    # 计算相关系数（使用 numpy）
    correlation_matrix = np.corrcoef(num_bit_errors_per_sample, angle_errors_deg)
    correlation = correlation_matrix[0, 1]
    
    ax6.set_xlabel('Number of Bit Errors per Sample', fontsize=12)
    ax6.set_ylabel('Angle Error (deg)', fontsize=12)
    ax6.set_title(f'Bit Errors vs Angle Error\n(Correlation: {correlation:.3f})', fontsize=14)
    ax6.legend(fontsize=10, loc='upper left')
    ax6.grid(linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax6, label='Bit Errors')
    
    # 8.7 每个位错误对角度误差的影响
    # 计算每个位单独错误时的平均角度误差影响
    bit_impact_on_angle = np.zeros(output_bits)
    for bit_idx in range(output_bits):
        # 找到该位错误的样本
        bit_error_mask = bit_errors[:, bit_idx] == 1
        if bit_error_mask.sum() > 0:
            # 计算这些样本的平均角度误差
            bit_impact_on_angle[bit_idx] = angle_errors_deg[bit_error_mask].mean()
        else:
            bit_impact_on_angle[bit_idx] = 0
    
    ax7 = plt.subplot(3, 3, 7)
    bars2 = ax7.bar(range(output_bits), bit_impact_on_angle, 
                   color='crimson', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax7.set_xlabel('Bit Index', fontsize=12)
    ax7.set_ylabel('Mean Angle Error (deg)', fontsize=12)
    ax7.set_title('Bit Error Impact on Angle\n(When bit is wrong)', fontsize=14)
    ax7.grid(axis='y', linestyle='--', alpha=0.5)
    ax7.set_xticks(range(0, output_bits, max(1, output_bits // 10)))
    
    # 添加数值标签（只显示影响较大的）
    for i, (bar, impact) in enumerate(zip(bars2, bit_impact_on_angle)):
        if impact > np.percentile(bit_impact_on_angle[bit_impact_on_angle > 0], 50) if (bit_impact_on_angle > 0).sum() > 0 else 0:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{impact:.1f}°', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 更新统计表格，添加角度影响信息
    df_stats['Angle Impact (deg)'] = bit_impact_on_angle.tolist()
    df_stats.to_csv(stats_csv, index=False, float_format='%.6f')
    
    # 9. 创建每步的编码对比表格（优化格式：用二进制字符串展示）
    n_samples_to_save = min(200, len(gt_heading))  # 保存前200个样本的详细对比
    
    # 将二进制数组转换为字符串
    def binary_array_to_string(binary_array):
        return ''.join([str(int(b)) for b in binary_array])
    
    step_comparison_data = []
    for i in range(n_samples_to_save):
        gt_binary_str = binary_array_to_string(gt_binary_hard[i])
        pred_binary_str = binary_array_to_string(pred_binary_hard[i])
        
        # 标记错误位（用^标记错误的位置）
        error_markers = []
        for j in range(output_bits):
            if gt_binary_hard[i, j] != pred_binary_hard[i, j]:
                error_markers.append('^')
            else:
                error_markers.append(' ')
        error_marker_str = ''.join(error_markers)
        
        step_comparison_data.append({
            'Step': i,
            'GT_Heading (deg)': f"{np.degrees(gt_heading[i]):.6f}",
            'Pred_Heading (deg)': f"{np.degrees(pred_heading[i]):.6f}",
            'Angle_Error (deg)': f"{angle_errors_deg[i]:.6f}",
            'Num_Bit_Errors': int(num_bit_errors_per_sample[i]),
            'GT_Encoding': gt_binary_str,
            'Pred_Encoding': pred_binary_str,
            'Error_Markers': error_marker_str
        })
    
    df_step_comparison = pd.DataFrame(step_comparison_data)
    step_comparison_csv = os.path.join(output_dir, f"{file_prefix}_step_encoding_comparison.csv")
    df_step_comparison.to_csv(step_comparison_csv, index=False)
    
    # 10. 单比特错误影响分析（只错单个位时的角度误差）
    single_bit_error_impact = {}
    single_bit_error_stats = {
        'Bit_Index': [],
        'Error_Type': [],  # '0->1' or '1->0'
        'Num_Samples': [],
        'Mean_Angle_Error (deg)': [],
        'Std_Angle_Error (deg)': [],
        'Max_Angle_Error (deg)': []
    }
    
    for bit_idx in range(output_bits):
        # 找到只错这一位的样本（其他位都正确）
        single_error_mask = (num_bit_errors_per_sample == 1) & (bit_errors[:, bit_idx] == 1)
        
        if single_error_mask.sum() > 0:
            single_error_angles = angle_errors_deg[single_error_mask]
            single_bit_error_impact[bit_idx] = {
                'mean': single_error_angles.mean(),
                'std': single_error_angles.std(),
                'max': single_error_angles.max(),
                'count': single_error_mask.sum()
            }
            
            # 分析错误类型（0->1 还是 1->0）
            gt_bits_when_error = gt_binary_hard[single_error_mask, bit_idx]
            pred_bits_when_error = pred_binary_hard[single_error_mask, bit_idx]
            
            error_0_to_1 = ((gt_bits_when_error == 0) & (pred_bits_when_error == 1)).sum()
            error_1_to_0 = ((gt_bits_when_error == 1) & (pred_bits_when_error == 0)).sum()
            
            if error_0_to_1 > 0:
                mask_0_to_1 = single_error_mask.copy()
                mask_0_to_1[single_error_mask] = (gt_binary_hard[single_error_mask, bit_idx] == 0) & \
                                                  (pred_binary_hard[single_error_mask, bit_idx] == 1)
                angles_0_to_1 = angle_errors_deg[mask_0_to_1]
                single_bit_error_stats['Bit_Index'].append(bit_idx)
                single_bit_error_stats['Error_Type'].append('0->1')
                single_bit_error_stats['Num_Samples'].append(error_0_to_1)
                single_bit_error_stats['Mean_Angle_Error (deg)'].append(angles_0_to_1.mean())
                single_bit_error_stats['Std_Angle_Error (deg)'].append(angles_0_to_1.std())
                single_bit_error_stats['Max_Angle_Error (deg)'].append(angles_0_to_1.max())
            
            if error_1_to_0 > 0:
                mask_1_to_0 = single_error_mask.copy()
                mask_1_to_0[single_error_mask] = (gt_binary_hard[single_error_mask, bit_idx] == 1) & \
                                                  (pred_binary_hard[single_error_mask, bit_idx] == 0)
                angles_1_to_0 = angle_errors_deg[mask_1_to_0]
                single_bit_error_stats['Bit_Index'].append(bit_idx)
                single_bit_error_stats['Error_Type'].append('1->0')
                single_bit_error_stats['Num_Samples'].append(error_1_to_0)
                single_bit_error_stats['Mean_Angle_Error (deg)'].append(angles_1_to_0.mean())
                single_bit_error_stats['Std_Angle_Error (deg)'].append(angles_1_to_0.std())
                single_bit_error_stats['Max_Angle_Error (deg)'].append(angles_1_to_0.max())
        else:
            single_bit_error_impact[bit_idx] = {
                'mean': 0,
                'std': 0,
                'max': 0,
                'count': 0
            }
    
    df_single_bit_impact = pd.DataFrame(single_bit_error_stats)
    single_bit_impact_csv = os.path.join(output_dir, f"{file_prefix}_single_bit_error_impact.csv")
    df_single_bit_impact.to_csv(single_bit_impact_csv, index=False, float_format='%.6f')
    
    # 8.8 单比特错误影响（添加到主图）
    # 提取单比特错误的平均角度误差
    single_bit_mean_errors = [single_bit_error_impact[i]['mean'] for i in range(output_bits)]
    single_bit_counts = [single_bit_error_impact[i]['count'] for i in range(output_bits)]
    
    ax8 = plt.subplot(3, 3, 8)
    bars3 = ax8.bar(range(output_bits), single_bit_mean_errors, 
                   color='darkorange', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax8.set_xlabel('Bit Index', fontsize=12)
    ax8.set_ylabel('Mean Angle Error (deg)', fontsize=12)
    ax8.set_title('Single Bit Error Impact\n(Exactly 1 bit error)', fontsize=14)
    ax8.grid(axis='y', linestyle='--', alpha=0.5)
    ax8.set_xticks(range(0, output_bits, max(1, output_bits // 10)))
    
    # 添加样本数量标签（只显示有数据的）
    for i, (bar, count) in enumerate(zip(bars3, single_bit_counts)):
        if count > 0:
            height = bar.get_height()
            if height > 0:
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}°\n(n={count})', ha='center', va='bottom', fontsize=7)
    
    # 8.9 单比特错误样本数量
    ax9 = plt.subplot(3, 3, 9)
    bars4 = ax9.bar(range(output_bits), single_bit_counts, 
                    color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax9.set_xlabel('Bit Index', fontsize=12)
    ax9.set_ylabel('Number of Samples', fontsize=12)
    ax9.set_title('Single Bit Error Count\n(Per bit position)', fontsize=14)
    ax9.grid(axis='y', linestyle='--', alpha=0.5)
    ax9.set_xticks(range(0, output_bits, max(1, output_bits // 10)))
    
    # 添加数值标签（只显示有数据的）
    for i, (bar, count) in enumerate(zip(bars4, single_bit_counts)):
        if count > 0:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(count)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{file_prefix}_encoding_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 11. 打印统计摘要
    print(f"\n  📊 编码错误统计:")
    print(f"    总样本数: {len(gt_heading)}")
    print(f"    完全正确样本: {perfect_samples} ({perfect_rate*100:.2f}%)")
    print(f"    平均每样本错误位数: {num_bit_errors_per_sample.mean():.2f}")
    print(f"    平均位错误率: {bit_error_rates.mean()*100:.2f}%")
    print(f"    最大位错误率: {bit_error_rates.max()*100:.2f}% (Bit {bit_error_rates.argmax()})")
    print(f"    最小位错误率: {bit_error_rates.min()*100:.2f}% (Bit {bit_error_rates.argmin()})")
    print(f"\n  📐 角度误差影响分析:")
    print(f"    错误位数与角度误差相关系数: {correlation:.3f}")
    max_impact_bit = np.argmax(bit_impact_on_angle)
    print(f"    影响最大的位: Bit {max_impact_bit} (平均角度误差: {bit_impact_on_angle[max_impact_bit]:.2f}°)")
    print(f"    平均角度误差（当有位错误时）: {angle_errors_deg.mean():.2f}°")
    print(f"    平均角度误差（当无错误时）: {angle_errors_deg[num_bit_errors_per_sample == 0].mean():.2f}°" 
          if (num_bit_errors_per_sample == 0).sum() > 0 else "    (无完全正确的样本)")
    
    # 单比特错误影响摘要
    print(f"\n  🔍 单比特错误影响分析:")
    for bit_idx in range(output_bits):
        if single_bit_error_impact[bit_idx]['count'] > 0:
            print(f"    Bit {bit_idx}: {single_bit_error_impact[bit_idx]['count']} 个样本, "
                  f"平均角度误差: {single_bit_error_impact[bit_idx]['mean']:.2f}°, "
                  f"最大: {single_bit_error_impact[bit_idx]['max']:.2f}°")
    
    print(f"\n  💾 文件保存:")
    print(f"    位错误率统计表格: {stats_csv}")
    print(f"    错误分布表格: {error_dist_csv}")
    print(f"    每步编码对比表格: {step_comparison_csv}")
    print(f"    单比特错误影响表格: {single_bit_impact_csv}")
    print(f"    编码分析图（合并所有可视化）: {output_path}")
    
    return df_stats, df_error_dist


def plot_trajectory_with_quiver(positions, headings_gt, headings_pred, step_interval=20, arrow_length=0.4, output_file=None):
    """
    绘制带有【真值 vs 预测】双箭头的轨迹图 (Quiver Plot)
    
    Args:
        positions: (N, 2) 轨迹坐标 (x, y)
        headings_gt: (N,) 真值绝对航向角 (弧度)
        headings_pred: (N,) 预测绝对航向角 (弧度)
        step_interval: 采样间隔
        arrow_length: 箭头长度(米)
        output_file: 保存路径
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import numpy as np
    import matplotlib.lines as mlines

    x = positions[:, 0]
    y = positions[:, 1]
    
    # 设置画布
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.set_facecolor('white')
    
    # 1. 绘制轨迹线 (使用预测航向着色，或者简单的黑色)
    # 为了突出箭头对比，轨迹线建议用简单的灰色或黑色，不要抢眼
    ax.plot(x, y, 'k-', linewidth=1.5, alpha=0.3, label='Trajectory', zorder=1)

    # 2. 绘制双箭头 (真值=绿, 预测=红)
    indices = np.arange(0, len(positions), step_interval)
    # 过滤掉越界索引
    max_idx = min(len(headings_gt), len(headings_pred), len(positions))
    indices = indices[indices < max_idx]
    
    # 箭头样式参数 (调小了比例，看起来更精致)
    hw = arrow_length * 0.25  # head width
    hl = arrow_length * 0.35  # head length
    
    for i in indices:
        px, py = x[i], y[i]
        
        # --- 画真值箭头 (绿色) ---
        h_gt = headings_gt[i]
        dx_gt = arrow_length * np.cos(h_gt)
        dy_gt = arrow_length * np.sin(h_gt)
        
        ax.arrow(px, py, dx_gt, dy_gt, 
                 head_width=hw, head_length=hl, length_includes_head=True,
                 fc='#2ecc71', ec='#27ae60',  # 亮绿填充，深绿边
                 alpha=0.6, linewidth=0.8, zorder=5)

        # --- 画预测箭头 (红色) ---
        h_pred = headings_pred[i]
        dx_pred = arrow_length * np.cos(h_pred)
        dy_pred = arrow_length * np.sin(h_pred)
        
        ax.arrow(px, py, dx_pred, dy_pred, 
                 head_width=hw, head_length=hl, length_includes_head=True,
                 fc='#e74c3c', ec='#c0392b',  # 亮红填充，深红边
                 alpha=0.8, linewidth=0.8, zorder=10) # 预测在最上层

    # 3. 装饰图表
    ax.set_title(f'Heading Comparison: GT (Green) vs Pred (Red)\n(Arrow every {step_interval} steps)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.axis('equal')
    ax.grid(True, linestyle='--', color='gray', alpha=0.3)
    
    # 起终点
    ax.scatter(x[0], y[0], c='black', s=100, marker='o', label='Start', zorder=11)
    ax.scatter(x[-1], y[-1], c='black', s=100, marker='X', label='End', zorder=11)
    
    # 自定义图例 (因为 arrow 没法很好地自动生成图例)
    legend_handles = [
        mlines.Line2D([], [], color='#2ecc71', marker='>', markersize=10, linestyle='None', label='Ground Truth Heading'),
        mlines.Line2D([], [], color='#e74c3c', marker='>', markersize=10, linestyle='None', label='Predicted Heading'),
        mlines.Line2D([], [], color='black', linewidth=1.5, alpha=0.3, label='Trajectory')
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=True, fancybox=True, shadow=True)

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    plt.close()


def plot_trajectory_turn_error_quiver(positions, headings_pred, headings_local_truth, step_interval=20, arrow_length=0.4, output_file=None):
    """
    绘制【预测 vs 瞬时转向修正】双箭头轨迹图
    用于发现引起轨迹偏差的突变点。
    
    Args:
        positions: (N, 2) 轨迹坐标 (x, y)
        headings_pred: (N,) 预测绝对航向角 (红色箭头)
        headings_local_truth: (N,) 局部真值/瞬时修正航向角 (紫色箭头)
                              计算公式: Pred_Heading + (GT_Turn - Pred_Turn)
        step_interval: 采样间隔
        arrow_length: 箭头长度(米)
        output_file: 保存路径
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.lines as mlines

    x = positions[:, 0]
    y = positions[:, 1]
    
    # 设置画布
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.set_facecolor('white')
    
    # 1. 绘制轨迹线 (淡灰色底图，不抢眼)
    ax.plot(x, y, 'k-', linewidth=1.5, alpha=0.2, label='Trajectory', zorder=1)

    # 2. 绘制双箭头
    indices = np.arange(0, len(positions), step_interval)
    max_idx = min(len(headings_local_truth), len(headings_pred), len(positions))
    indices = indices[indices < max_idx]
    
    # 箭头样式
    hw = arrow_length * 0.25
    hl = arrow_length * 0.35
    
    for i in indices:
        px, py = x[i], y[i]

        # --- A. 画预测箭头 (红色，代表模型当前的实际选择) ---
        h_pred = headings_pred[i]
        dx_pred = arrow_length * np.cos(h_pred)
        dy_pred = arrow_length * np.sin(h_pred)
        
        ax.arrow(px, py, dx_pred, dy_pred, 
                 head_width=hw, head_length=hl, length_includes_head=True,
                 fc='#e74c3c', ec='#c0392b',  # 亮红填充
                 alpha=0.9, linewidth=0.8, zorder=5)

        # --- B. 画瞬时修正箭头 (紫色，代表"如果这步转对了应该朝哪") ---
        h_local = headings_local_truth[i]
        dx_local = arrow_length * np.cos(h_local)
        dy_local = arrow_length * np.sin(h_local)
        
        # 只有当两者有明显夹角时，紫色箭头才明显分叉，直观展示误差
        ax.arrow(px, py, dx_local, dy_local, 
                 head_width=hw, head_length=hl, length_includes_head=True,
                 fc='#2ecc71', ec='#27ae60',  # 亮绿填充，深绿边
                 alpha=0.8, linewidth=0.8, zorder=10)

    # 3. 装饰
    ax.set_title(f'Instantaneous Turn Error Analysis\nRed=Prediction, Purple=Ideal Heading (if turn was correct)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.axis('equal')
    ax.grid(True, linestyle='--', color='gray', alpha=0.3)
    
    # 起终点
    ax.scatter(x[0], y[0], c='green', s=100, marker='o', label='Start', zorder=11, edgecolors='k')
    ax.scatter(x[-1], y[-1], c='black', s=100, marker='X', label='End', zorder=11, edgecolors='k')
    
    # 自定义图例
    legend_handles = [
        mlines.Line2D([], [], color='#e74c3c', marker='>', markersize=10, linestyle='None', label='Predicted Heading'),
        mlines.Line2D([], [], color='#9b59b6', marker='>', markersize=10, linestyle='None', label='Local Ideal Heading (Turn Corrected)'),
        mlines.Line2D([], [], color='black', linewidth=1.5, alpha=0.2, label='Trajectory')
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=True, shadow=True)

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    plt.close()
