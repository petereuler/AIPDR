import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset_RIDI import load_ridi_raw, window_dataset as ridi_window, yaw_from_quat
from models.pose_net import PoseNetTransformer, quat_to_rotmat
from utils.results import reconstruct_from_absolute_angles

# ======= 基础工具函数 =======

def rotmat_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    r00, r10, r20 = R[:, 0, 0], R[:, 1, 0], R[:, 2, 0]
    r21, r22 = R[:, 2, 1], R[:, 2, 2]
    pitch = torch.asin(torch.clamp(-r20, -1.0, 1.0))
    roll = torch.atan2(r21, r22)
    yaw = torch.atan2(r10, r00)
    return torch.stack([roll, pitch, yaw], dim=1)

def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi

def unwrap_rad(x):
    x = np.array(x, dtype=np.float32).reshape(-1)
    return np.unwrap(x)

def load_ridi_sequence(ridi_root: str, seq_name: str, window_size: int, stride: int):
    seq_dir = os.path.join(ridi_root, "data", seq_name)
    if not os.path.isdir(seq_dir):
        return None
    
    # 1. 加载原始数据
    gyro, acc, pos3d, ori = load_ridi_raw(seq_dir)
    
    # 2. 切分窗口
    try:
        # y_rel 就是 stride 期间的相对旋转
        [gx, ax], [dl, _, abs_h, y_rel], init_pos, _ = ridi_window(
            gyro, acc, pos3d, ori,
            mode="2d", window_size=window_size, stride=stride,
            filter_window=20, smooth_heading=True, heading_sigma=1.5,
            return_abs_heading=True, return_rel_ori=True, align_heading_to_init_pose=False
        )
    except Exception as e:
        print(f"Error processing {seq_name}: {e}")
        return None

    x = np.concatenate([gx, ax], axis=-1)
    
    # 对齐长度
    # 手动计算对应的索引，确保真值对齐
    max_start = gyro.shape[0] - window_size - 1
    b_indices = []
    for idx in range(0, max_start, stride):
        b = idx + window_size // 2 + stride // 2
        b = max(0, min(b, len(ori) - 1))
        b_indices.append(b)
    
    min_len = min(len(y_rel), len(b_indices))
    if min_len == 0:
        return None

    x = x[:min_len]
    y_rel = y_rel[:min_len]
    abs_h = abs_h[:min_len]
    dl = dl[:min_len]
    ori_seq = ori[b_indices[:min_len]]

    return x, y_rel, abs_h, dl, init_pos, ori_seq

# ======= 三大绘图函数 =======

def plot_relative_euler(R_pred, R_gt, out_dir, seq_name):
    """图1: 相对欧拉角对比 (检查 PoseNet 单步预测能力)"""
    euler_pred = rotmat_to_euler_xyz(R_pred).cpu().numpy()
    euler_gt = rotmat_to_euler_xyz(R_gt).cpu().numpy()
    
    t = np.arange(euler_pred.shape[0])
    deg = 180.0 / np.pi
    labels = ["Roll", "Pitch", "Yaw"]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axes[i].plot(t, euler_gt[:, i] * deg, label="GT (Stride)", linewidth=2, alpha=0.6)
        axes[i].plot(t, euler_pred[:, i] * deg, label="Pred (Stride)", linestyle="--", alpha=0.8)
        axes[i].set_ylabel(f"{labels[i]} (deg)")
        axes[i].grid(True, alpha=0.3)
        if i == 0: axes[i].legend(loc="upper right")
    
    axes[-1].set_xlabel("Window Index (Stride Steps)")
    fig.suptitle(f"Relative Rotation (Per Stride): {seq_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{seq_name}_1_rel.png"), dpi=100)
    plt.close()

def plot_trajectory_comparison(init_pos, dl, yaw_pred_abs, yaw_gt_motion, out_dir, seq_name):
    """图2: 轨迹重建对比 (Pred Yaw + GT Len vs GT Traj)"""
    traj_pred = reconstruct_from_absolute_angles(init_pos, dl, yaw_pred_abs)
    traj_gt = reconstruct_from_absolute_angles(init_pos, dl, yaw_gt_motion)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(traj_gt[:, 0], traj_gt[:, 1], label="GT Trajectory", linewidth=2, color="black", alpha=0.6)
    ax.plot(traj_pred[:, 0], traj_pred[:, 1], label="Pred Yaw + GT Len", linewidth=1.5, color="orange", linestyle="--")
    
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"Trajectory Reconstruction: {seq_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{seq_name}_2_traj.png"), dpi=100)
    plt.close()

def plot_heading_analysis(yaw_phone_gt, yaw_phone_pred, heading_motion_gt, out_dir, seq_name):
    """图3: 航向角分析 (Accumulated Pose Yaw vs Motion Heading)"""
    t = np.arange(len(yaw_phone_gt))
    deg = 180.0 / np.pi
    
    # 解缠绕 (Unwrap) 以观察长期漂移趋势
    u_phone_gt = unwrap_rad(yaw_phone_gt)
    u_phone_pred = unwrap_rad(yaw_phone_pred)
    u_motion_gt = unwrap_rad(heading_motion_gt)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. 真实运动方向 (走路方向)
    ax.plot(t, u_motion_gt * deg, label="Motion Heading (GT)", color="#1f77b4", linestyle="--", linewidth=2)
    # 2. 真实手机姿态 (手机朝向)
    ax.plot(t, u_phone_gt * deg, label="Phone Yaw (GT)", color="#2ca02c", alpha=0.6, linewidth=2)
    # 3. 预测手机姿态 (PoseNet 积分结果)
    ax.plot(t, u_phone_pred * deg, label="Phone Yaw (Pred Accumulated)", color="#ff7f0e", alpha=0.8, linewidth=1.5)
    
    ax.set_ylabel("Accumulated Yaw (deg)")
    ax.set_xlabel("Window Index")
    ax.set_title(f"Heading Analysis: {seq_name}\n(Drift Check)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{seq_name}_3_head.png"), dpi=100)
    plt.close()


def plot_absolute_euler(R_pred, R_gt, out_dir, seq_name):
    """图4: 绝对欧拉角对比 (累积姿态, 对齐初始值)"""
    euler_pred = rotmat_to_euler_xyz(R_pred).cpu().numpy()
    euler_gt = rotmat_to_euler_xyz(R_gt).cpu().numpy()

    if euler_pred.shape[0] == 0 or euler_gt.shape[0] == 0:
        return

    euler_pred = wrap_angle(euler_pred - euler_pred[0:1, :])
    euler_gt = wrap_angle(euler_gt - euler_gt[0:1, :])

    t = np.arange(euler_pred.shape[0])
    deg = 180.0 / np.pi
    labels = ["Roll", "Pitch", "Yaw"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axes[i].plot(t, euler_gt[:, i] * deg, label="GT (Absolute)", linewidth=2, alpha=0.6)
        axes[i].plot(t, euler_pred[:, i] * deg, label="Pred (Accumulated)", linestyle="--", alpha=0.8)
        axes[i].set_ylabel(f"{labels[i]} (deg)")
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc="upper right")

    axes[-1].set_xlabel("Window Index (Stride Steps)")
    fig.suptitle(f"Absolute Rotation (Accumulated): {seq_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{seq_name}_4_abs.png"), dpi=100)
    plt.close()


# ======= 主流程 =======

def main():
    # 配置
    ridi_root = "/home/admin407/code/zyshe/NavCorrector/RIDI"
    ckpt_path = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls/pose_net.pth"
    out_dir = "/home/admin407/code/zyshe/NavCorrector/output_all"
    os.makedirs(out_dir, exist_ok=True)
    
    # 参数必须与训练一致
    WINDOW_SIZE = 64
    STRIDE = 64
    pose_output_mode = "quat"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 加载模型
    print("Loading PoseNet...")
    pose_net = PoseNetTransformer(
        imu_dim=6,
        d_model=128,
        output_mode=pose_output_mode,
    ).to(device)
    if os.path.exists(ckpt_path):
        pose_net.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Checkpoint loaded.")
    else:
        print("Error: Checkpoint not found! Using random weights.")
    pose_net.eval()

    # 2. 获取测试列表
    test_list_path = os.path.join(ridi_root, "data", "list_test_publish_v2.txt")
    if not os.path.exists(test_list_path):
        print(f"Test list not found at {test_list_path}")
        return

    with open(test_list_path, "r") as f:
        seq_names = [line.strip().split(",")[0] for line in f if line.strip()]
    
    print(f"Found {len(seq_names)} test sequences.")
    
    # 统计数据
    global_z_stats = []
    
    # 3. 循环处理所有文件
    for seq_name in tqdm(seq_names, desc="Processing"):
        data = load_ridi_sequence(ridi_root, seq_name, window_size=WINDOW_SIZE, stride=STRIDE)
        if data is None:
            continue
            
        x, y_rel, abs_h_gt, dl, init_pos, ori_gt_seq = data
        xb = torch.tensor(x, dtype=torch.float32, device=device)
        
        # 推理
        with torch.no_grad():
            R_pred_rel = pose_net(xb) # [B, 3, 3] Stride 旋转
        
        # 准备数据: 相对旋转真值
        q_rel_gt = torch.tensor(y_rel, dtype=torch.float32, device=device)
        R_gt_rel = quat_to_rotmat(q_rel_gt)
        
        # --- 重力对齐统计 (Gravity Check) ---
        acc_local = xb[:, :, 3:6]
        # 使用预测的 stride 旋转可能会有一点偏差，但大体能看出来
        # 这里更严谨应该用 Anchor Mode 的绝对姿态，但简单起见用相对旋转看均值
        acc_global_pred = torch.matmul(acc_local, R_pred_rel.transpose(1, 2))
        z_pred = acc_global_pred.mean(dim=1)[:, 2].cpu().numpy()
        global_z_stats.append(np.std(z_pred))
        
        # --- 积分绝对姿态 (Accumulation) ---
        # 逻辑：R_next = R_curr @ R_stride_pred
        # 这是一个连续的积分过程，会展现累积漂移
        
        init_q = ori_gt_seq[0]
        init_R = quat_to_rotmat(torch.tensor(init_q[None, :], dtype=torch.float32)).to(device).squeeze(0)
        
        R_abs_list = []
        curr_R = init_R
        for i in range(len(R_pred_rel)):
            # 核心累积行：连续左乘（假设 PoseNet 预测的是 Local Delta）
            # 或者右乘？trainc.py 里是 R_anchor @ R_delta，意味着 R_delta 是 Local (Body Frame)
            curr_R = curr_R @ R_pred_rel[i]
            R_abs_list.append(curr_R)
        R_abs_tensor = torch.stack(R_abs_list, dim=0)
        
        yaw_phone_pred_raw = rotmat_to_euler_xyz(R_abs_tensor)[:, 2].cpu().numpy()
        yaw_phone_gt_raw = yaw_from_quat(ori_gt_seq).flatten()
        heading_motion_gt_raw = abs_h_gt.flatten()

        # 不对齐航向角，使用绝对姿态与原始运动航向
        yaw_pred_for_traj = wrap_angle(yaw_phone_pred_raw)
        yaw_phone_pred = wrap_angle(yaw_phone_pred_raw)
        yaw_phone_gt = wrap_angle(yaw_phone_gt_raw)
        heading_motion_gt = wrap_angle(heading_motion_gt_raw)
        
        # === 生成三张图 ===
        plot_relative_euler(R_pred_rel, R_gt_rel, out_dir, seq_name)
        plot_trajectory_comparison(init_pos, dl, yaw_pred_for_traj, heading_motion_gt_raw, out_dir, seq_name)
        plot_heading_analysis(yaw_phone_gt, yaw_phone_pred, heading_motion_gt, out_dir, seq_name)
        R_abs_gt = quat_to_rotmat(torch.tensor(ori_gt_seq, dtype=torch.float32, device=device))
        plot_absolute_euler(R_abs_tensor, R_abs_gt, out_dir, seq_name)

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print(f"Output Directory: {out_dir}")
    print(f"Average Gravity Z Std: {np.mean(global_z_stats):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
