import os
import sys
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset_RIDI import load_ridi_raw, window_dataset as ridi_window
from utils.results import reconstruct_from_absolute_angles
from models.pose_net import PoseNetTransformer, quat_to_rotmat


def rotmat_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw) in XYZ order.
    R: (B, 3, 3)
    returns: (B, 3) in radians
    """
    r00 = R[:, 0, 0]
    r10 = R[:, 1, 0]
    r20 = R[:, 2, 0]
    r21 = R[:, 2, 1]
    r22 = R[:, 2, 2]

    pitch = torch.asin(torch.clamp(-r20, -1.0, 1.0))
    roll = torch.atan2(r21, r22)
    yaw = torch.atan2(r10, r00)
    return torch.stack([roll, pitch, yaw], dim=1)


def yaw_from_quat(q: np.ndarray) -> np.ndarray:
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def load_ridi_sequence(ridi_root: str, seq_name: str, window_size: int, stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq_dir = os.path.join(ridi_root, "data", seq_name)
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Sequence not found: {seq_dir}")
    gyro, acc, pos3d, ori = load_ridi_raw(seq_dir)
    [gx, ax], [dl, _, abs_h, y_rel], init_pos, _ = ridi_window(
        gyro, acc, pos3d, ori,
        mode="2d",
        window_size=window_size,
        stride=stride,
        filter_window=20,
        smooth_heading=True,
        heading_sigma=1.5,
        smooth_length=False,
        length_sigma=1.0,
        return_abs_heading=True,
        return_rel_ori=True,
        align_heading_to_init_pose=True,
    )
    x = np.concatenate([gx, ax], axis=-1)
    return x, y_rel, abs_h, dl, init_pos, ori


def pick_first_test_sequence(ridi_root: str) -> str:
    test_list = os.path.join(ridi_root, "data", "list_test_publish_v2.txt")
    if not os.path.exists(test_list):
        raise FileNotFoundError(f"Missing test list: {test_list}")
    with open(test_list, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return line.split(",")[0]
    raise RuntimeError("No valid entries found in test list.")


def main():
    ridi_root = "/home/admin407/code/zyshe/NavCorrector/RIDI"
    ckpt = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls/pose_net.pth"
    out_dir = "/home/admin407/code/zyshe/NavCorrector/output"
    max_n = 2000
    seq = ""
    window_size = 320
    stride = 64

    seq = seq or pick_first_test_sequence(ridi_root)
    x, y_rel, abs_h, dl, init_pos, ori = load_ridi_sequence(ridi_root, seq, window_size, stride)
    if x.shape[0] == 0:
        raise RuntimeError("No windows produced. Check window_size/stride.")

    x = x[:max_n]
    y_rel = y_rel[:max_n]
    abs_h = abs_h[:max_n]
    dl = dl[:max_n]

    # Sanity check: label self-MAE should be ~0
    diff = abs_h - abs_h
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    label_mae = np.degrees(np.mean(np.abs(diff)))
    print(f"[Label Check] abs_heading_gt self MAE: {label_mae:.6f} deg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_net = PoseNetTransformer(imu_dim=6, d_model=128, nhead=4, num_layers=2, dim_feedforward=256).to(device)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"PoseNet checkpoint not found: {ckpt}")
    pose_net.load_state_dict(torch.load(ckpt, map_location=device))
    pose_net.eval()

    xb = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        R_pred = pose_net(xb)

    q_rel = torch.tensor(y_rel, dtype=torch.float32, device=device)
    R_gt = quat_to_rotmat(q_rel)

    euler_pred = rotmat_to_euler_xyz(R_pred).cpu().numpy()
    euler_gt = rotmat_to_euler_xyz(R_gt).cpu().numpy()

    # Gravity alignment sanity: rotate local acc by predicted pose
    acc_local = xb[:, :, 3:6]
    R_pred_t = R_pred.transpose(1, 2)
    acc_global_pred = torch.matmul(acc_local, R_pred_t)
    acc_global_pred_mean = acc_global_pred.mean(dim=1).cpu().numpy()
    z_pred = acc_global_pred_mean[:, 2]
    print(f"[Gravity Check] pred acc_global z: mean={z_pred.mean():.4f}, std={z_pred.std():.4f}")

    R_gt_t = R_gt.transpose(1, 2)
    acc_global_gt = torch.matmul(acc_local, R_gt_t)
    acc_global_gt_mean = acc_global_gt.mean(dim=1).cpu().numpy()
    z_gt = acc_global_gt_mean[:, 2]
    print(f"[Gravity Check] gt   acc_global z: mean={z_gt.mean():.4f}, std={z_gt.std():.4f}")

    t = np.arange(euler_pred.shape[0])
    deg = 180.0 / np.pi
    labels = ["roll", "pitch", "yaw"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axes[i].plot(t, euler_gt[:, i] * deg, label="gt")
        axes[i].plot(t, euler_pred[:, i] * deg, label="pred", alpha=0.8)
        axes[i].set_ylabel(labels[i] + " (deg)")
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend()
    axes[-1].set_xlabel("window index")
    fig.suptitle(f"PoseNet Relative Euler: {seq}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pose_net_euler_{seq}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    # Absolute heading vs accumulated yaw
    gyro, acc, _, _ = load_ridi_raw(os.path.join(ridi_root, "data", seq))
    max_start = gyro.shape[0] - window_size - 1
    b_indices = []
    for idx in range(0, max_start, stride):
        b = idx + window_size // 2 + stride // 2
        b = max(0, min(b, len(ori) - 1))
        b_indices.append(b)
    b_indices = np.array(b_indices, dtype=np.int64)
    if b_indices.size > 0:
        b_indices = b_indices[:abs_h.shape[0]]
        q_seq = ori[b_indices].astype(np.float32)
        yaw_ori = yaw_from_quat(q_seq)
        R_ori = quat_to_rotmat(torch.tensor(q_seq, dtype=torch.float32))
        yaw_ori_b2w = rotmat_to_euler_xyz(R_ori)[:, 2].cpu().numpy()
        yaw_ori_w2b = rotmat_to_euler_xyz(R_ori.transpose(1, 2))[:, 2].cpu().numpy()

        init_q = ori[b_indices[0]].astype(np.float32)
        init_rot = quat_to_rotmat(torch.tensor(init_q[None, :], dtype=torch.float32)).squeeze(0)
        yaw0_b2w = float(np.arctan2(init_rot[1, 0].item(), init_rot[0, 0].item()))
        R_abs = []
        current = init_rot
        for i in range(R_pred.size(0)):
            current = current @ R_pred[i].cpu()
            R_abs.append(current)
        R_abs = torch.stack(R_abs, dim=0)
        yaw_pred_abs = rotmat_to_euler_xyz(R_abs)[:, 2].cpu().numpy()
        abs_h = wrap_angle(abs_h - yaw0_b2w)
        yaw_pred_abs = wrap_angle(yaw_pred_abs - yaw0_b2w)
        yaw_ori = wrap_angle(yaw_ori - yaw0_b2w)
        yaw_ori_b2w = wrap_angle(yaw_ori_b2w - yaw0_b2w)
        yaw_ori_w2b = wrap_angle(yaw_ori_w2b - yaw0_b2w)

        n = min(len(dl), len(yaw_pred_abs))
        traj_pred_pose = reconstruct_from_absolute_angles(init_pos, dl[:n], yaw_pred_abs[:n])
        traj_gt_pose = reconstruct_from_absolute_angles(init_pos, dl[:n], abs_h[:n].reshape(-1))
        fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5))
        ax3.plot(traj_gt_pose[:, 0], traj_gt_pose[:, 1], linewidth=1.5, label="traj_gt_abs_heading")
        ax3.plot(traj_pred_pose[:, 0], traj_pred_pose[:, 1], linewidth=1.2, label="traj_pose_pred_yaw_gt_len")
        ax3.axis("equal")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        fig3.suptitle(f"Trajectory from PoseNet yaw + GT length: {seq}")
        out_path3 = os.path.join(out_dir, f"pose_net_traj_pose_pred_gt_len_{seq}.png")
        plt.tight_layout()
        plt.savefig(out_path3, dpi=150)
        print(f"Saved: {out_path3}")

    def unwrap_rad(x):
        x = np.array(x, dtype=np.float32).reshape(-1)
        return np.unwrap(x)

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    colors = {
        "abs_heading_gt": "#1f77b4",
        "yaw_pred_abs": "#ff7f0e",
        "yaw_gt_from_ori": "#2ca02c",
        "yaw_gt_b2w": "#d62728",
        "yaw_gt_w2b": "#9467bd",
    }
    abs_h_u = unwrap_rad(abs_h[:len(t)].flatten())
    yaw_pred_u = unwrap_rad(yaw_pred_abs[:len(t)])
    yaw_ori_u = unwrap_rad(yaw_ori[:len(t)])
    yaw_b2w_u = unwrap_rad(yaw_ori_b2w[:len(t)])
    yaw_w2b_u = unwrap_rad(yaw_ori_w2b[:len(t)])

    ax2.plot(t[:len(abs_h_u)], abs_h_u * deg,
             label="abs_heading_gt", color=colors["abs_heading_gt"])
    ax2.plot(t[:len(yaw_pred_u)], yaw_pred_u * deg,
             label="yaw_pred_abs", color=colors["yaw_pred_abs"], alpha=0.85)
    ax2.plot(t[:len(yaw_ori_u)], yaw_ori_u * deg,
             label="yaw_gt_from_ori", color=colors["yaw_gt_from_ori"], alpha=0.7)
    ax2.plot(t[:len(yaw_b2w_u)], yaw_b2w_u * deg,
             label="yaw_gt_b2w", color=colors["yaw_gt_b2w"], alpha=0.7)
    ax2.plot(t[:len(yaw_w2b_u)], yaw_w2b_u * deg,
             label="yaw_gt_w2b", color=colors["yaw_gt_w2b"], alpha=0.7)
    ax2.set_ylabel("yaw (deg)")
    ax2.set_xlabel("window index")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.suptitle(f"Absolute Heading vs Pose Yaw: {seq}")

    out_path2 = os.path.join(out_dir, f"pose_net_abs_yaw_{seq}.png")
    plt.tight_layout()
    plt.savefig(out_path2, dpi=150)
    print(f"Saved: {out_path2}")


if __name__ == "__main__":
    main()
