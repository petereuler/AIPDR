import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到模块搜索路径
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset_RIDI import load_ridi_raw, window_dataset as ridi_window
from data.dataset_OXIOD import (
    get_oxiod_predefined_split_pairs,
    load_oxiod_raw,
    window_dataset as oxiod_window,
)
from models.posenet import PoseNetTransformer, quat_to_rotmat
from utils.visualization import (
    plot_absolute_euler,
    plot_heading_analysis,
    plot_posenet_trajectory_comparison,
    plot_relative_euler,
)

# 可选: "RIDI" / "OXIOD"
DATASET = "RIDI"

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

def yaw_from_quat(q):
    q = np.array(q, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, 4)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def unwrap_rad(x):
    x = np.array(x, dtype=np.float32).reshape(-1)
    return np.unwrap(x)


def reconstruct_from_absolute_angles(init_pos, step_lengths, absolute_angles):
    traj = [init_pos]
    curr_pos = init_pos.copy()
    n = min(len(step_lengths), len(absolute_angles))
    for i in range(n):
        step_len = step_lengths[i]
        theta = absolute_angles[i]
        dx = step_len * np.cos(theta)
        dy = step_len * np.sin(theta)
        curr_pos[0] += dx
        curr_pos[1] += dy
        traj.append(curr_pos.copy())
    return np.array(traj)


def align_yaw_to_motion(yaw_seq: np.ndarray, motion_heading_seq: np.ndarray) -> np.ndarray:
    """
    用首帧把姿态航向对齐到运动航向，避免常数偏置导致的整轨旋转错位。
    """
    y = np.array(yaw_seq, dtype=np.float32).reshape(-1)
    m = np.array(motion_heading_seq, dtype=np.float32).reshape(-1)
    if len(y) == 0 or len(m) == 0:
        return y
    offset = m[0] - y[0]
    return wrap_angle(y + offset)

def load_ridi_sequence(ridi_root: str, seq_name: str, window_size: int, stride: int):
    seq_dir = os.path.join(ridi_root, "data", seq_name)
    if not os.path.isdir(seq_dir):
        return None
    
    # 1. 加载原始数据
    gyro, acc, pos_xyz, ori = load_ridi_raw(seq_dir)
    
    # 2. 切分窗口
    try:
        # y_rel 是窗口对应的相对旋转；ydp_world 用于重建运动航向。
        [gx, ax], [dl, y_rel, ydp_world], init_pos, _ = ridi_window(
            gyro, acc, pos_xyz, ori,
            window_size=window_size, stride=stride,
            filter_window=20,
            smooth_length=False, length_sigma=1.0,
            return_rel_ori=True,
            return_delta_p_world=True,
        )
    except Exception as e:
        print(f"Error processing {seq_name}: {e}")
        return None

    x = np.concatenate([gx, ax], axis=-1)
    
    # 手动计算窗口终点索引，确保姿态真值和窗口标签对齐。
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
    heading_motion = np.arctan2(ydp_world[:min_len, 1], ydp_world[:min_len, 0]).reshape(-1, 1)
    dl = dl[:min_len]
    ori_seq = ori[b_indices[:min_len]]

    return x, y_rel, heading_motion, dl, init_pos, ori_seq


def load_oxiod_sequence(imu_file: str, gt_file: str, window_size: int, stride: int):
    gyro, acc, pos_xyz, ori = load_oxiod_raw(imu_file, gt_file)
    try:
        [gx, ax], [dl, y_rel, ydp_world], init_pos, _ = oxiod_window(
            gyro, acc, pos_xyz, ori,
            window_size=window_size, stride=stride,
            filter_window=20,
            smooth_length=False, length_sigma=1.0,
            return_rel_ori=True,
            return_delta_p_world=True,
        )
    except Exception as e:
        print(f"Error processing {imu_file}: {e}")
        return None

    x = np.concatenate([gx, ax], axis=-1)

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
    heading_motion = np.arctan2(ydp_world[:min_len, 1], ydp_world[:min_len, 0]).reshape(-1, 1)
    dl = dl[:min_len]
    ori_seq = ori[b_indices[:min_len]]
    return x, y_rel, heading_motion, dl, init_pos, ori_seq

# ======= 主流程 =======

def main():
    # 配置
    dataset_name = os.getenv("DATASET", DATASET).upper()  # RIDI / OXIOD
    project_dir = os.path.dirname(os.path.abspath(__file__))
    ridi_root = os.path.join(project_dir, "RIDI")
    oxiod_root = os.path.join(project_dir, "OXIOD")
    ckpt_path = os.path.join(project_dir, "checkpoints", dataset_name.lower(), "posenet.pth")
    out_dir = os.path.join(project_dir, "output", "posenet", dataset_name.lower())
    os.makedirs(out_dir, exist_ok=True)
    
    # 这些参数需要和训练脚本保持一致。
    WINDOW_SIZE = 64
    STRIDE = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 加载 posenet
    print("Loading posenet...")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        posenet = PoseNetTransformer(
            imu_dim=6,
            d_model=128,
        ).to(device)
        posenet.load_state_dict(state)
        print("Checkpoint loaded.")
    else:
        posenet = PoseNetTransformer(
            imu_dim=6,
            d_model=128,
        ).to(device)
        print("Checkpoint loaded.")
    posenet.eval()

    # 2. 获取测试列表
    if dataset_name == "RIDI":
        test_list_path = os.path.join(ridi_root, "data", "list_test_publish_v2.txt")
        if not os.path.exists(test_list_path):
            print(f"Test list not found at {test_list_path}")
            return
        with open(test_list_path, "r") as f:
            seq_specs = [("RIDI", line.strip().split(",")[0]) for line in f if line.strip()]
    elif dataset_name == "OXIOD":
        seq_specs = [("OXIOD", name, imu, gt) for name, imu, gt in get_oxiod_predefined_split_pairs(oxiod_root, split="test", sensor="syn")]
    else:
        print(f"Unsupported DATASET={dataset_name}")
        return

    print(f"Found {len(seq_specs)} test sequences.")
    
    # 统计数据
    global_z_stats = []
    
    # 3. 循环处理所有序列
    for spec in tqdm(seq_specs, desc="Processing"):
        if spec[0] == "RIDI":
            seq_name = spec[1]
            data = load_ridi_sequence(ridi_root, seq_name, window_size=WINDOW_SIZE, stride=STRIDE)
        else:
            seq_name, imu_file, gt_file = spec[1], spec[2], spec[3]
            data = load_oxiod_sequence(imu_file, gt_file, window_size=WINDOW_SIZE, stride=STRIDE)
        if data is None:
            continue
            
        x, y_rel, heading_motion_gt, dl, init_pos, ori_gt_seq = data
        xb = torch.tensor(x, dtype=torch.float32, device=device)
        
        # 推理
        with torch.no_grad():
            q_rel_pred = posenet(xb)
            R_pred_rel = quat_to_rotmat(q_rel_pred)
        
        # 准备相对旋转真值
        q_rel_gt = torch.tensor(y_rel, dtype=torch.float32, device=device)
        R_gt_rel = quat_to_rotmat(q_rel_gt)
        
        # 统计预测相对旋转下的重力 z 轴波动。
        acc_local = xb[:, :, 3:6]
        acc_global_pred = torch.matmul(acc_local, R_pred_rel.transpose(1, 2))
        z_pred = acc_global_pred.mean(dim=1)[:, 2].cpu().numpy()
        global_z_stats.append(np.std(z_pred))
        
        # 累积相对旋转，得到整段序列的绝对姿态估计。
        
        init_q = ori_gt_seq[0]
        init_R = quat_to_rotmat(torch.tensor(init_q[None, :], dtype=torch.float32)).to(device).squeeze(0)
        
        R_abs_list = []
        curr_R = init_R
        for i in range(len(R_pred_rel)):
            # 当前约定下，相对旋转增量右乘到当前绝对姿态。
            curr_R = curr_R @ R_pred_rel[i]
            R_abs_list.append(curr_R)
        R_abs_tensor = torch.stack(R_abs_list, dim=0)
        
        yaw_phone_pred_raw = rotmat_to_euler_xyz(R_abs_tensor)[:, 2].cpu().numpy()
        yaw_phone_gt_raw = yaw_from_quat(ori_gt_seq).flatten()
        heading_motion_gt_raw = heading_motion_gt.flatten()

        # 轨迹图里先做一个常数航向对齐，避免固定偏置主导可视化效果。
        yaw_pred_for_traj = align_yaw_to_motion(yaw_phone_pred_raw, heading_motion_gt_raw)
        yaw_phone_pred = wrap_angle(yaw_phone_pred_raw)
        yaw_phone_gt = wrap_angle(yaw_phone_gt_raw)
        heading_motion_gt = wrap_angle(heading_motion_gt_raw)
        
        # 生成四张诊断图。
        safe_seq_name = seq_name.replace("/", "_").replace("\\", "_")
        plot_relative_euler(R_pred_rel, R_gt_rel, out_dir, safe_seq_name, rotmat_to_euler_xyz)
        plot_posenet_trajectory_comparison(
            init_pos, dl, yaw_pred_for_traj, heading_motion_gt_raw, out_dir, safe_seq_name, reconstruct_from_absolute_angles
        )
        plot_heading_analysis(yaw_phone_gt, yaw_phone_pred, heading_motion_gt, out_dir, safe_seq_name, unwrap_rad)
        R_abs_gt = quat_to_rotmat(torch.tensor(ori_gt_seq, dtype=torch.float32, device=device))
        plot_absolute_euler(R_abs_tensor, R_abs_gt, out_dir, safe_seq_name, rotmat_to_euler_xyz, wrap_angle)

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print(f"Output Directory: {out_dir}")
    print(f"Average Gravity Z Std: {np.mean(global_z_stats):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
