import os

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_comparison(traj_gt, traj_gt_xy, traj_pred, output_dir, base_name):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.plot(traj_gt_xy[:, 0], traj_gt_xy[:, 1], color="black", linewidth=1.8, label="GT")
    ax.plot(traj_pred[:, 0], traj_pred[:, 1], color="red", linewidth=1.2, alpha=0.85, label="Pred")
    ax.scatter(traj_gt_xy[0, 0], traj_gt_xy[0, 1], c="green", s=60, marker="o", label="Start")
    ax.scatter(traj_gt_xy[-1, 0], traj_gt_xy[-1, 1], c="purple", s=60, marker="X", label="End")
    ax.set_title(base_name)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_trajectory.png"), bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_dp(gt_dp, pred_dp, vis_num, output_path):
    t = np.arange(vis_num)
    cum_gt = np.cumsum(gt_dp[:vis_num], axis=0)
    cum_pred = np.cumsum(pred_dp[:vis_num], axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=150, sharex=True)
    labels = ["dx", "dy", "dz"]
    for i in range(3):
        axes[i].plot(t, cum_gt[:, i], label="GT", linewidth=1.2, color="black")
        axes[i].plot(t, cum_pred[:, i], label="Pred", linewidth=1.0, color="red", alpha=0.8)
        axes[i].set_ylabel(f"cum {labels[i]} (m)", fontsize=11)
        axes[i].grid(linestyle=":", alpha=0.6)
        if i == 0:
            axes[i].legend(fontsize=9, loc="upper right")
    axes[-1].set_xlabel("Window Index", fontsize=11)
    fig.suptitle("Cumulative dx/dy/dz", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_dp_series(gt_dp, pred_dp, vis_num, output_path):
    t = np.arange(vis_num)
    gt = gt_dp[:vis_num]
    pred = pred_dp[:vis_num]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=150, sharex=True)
    labels = ["dx", "dy", "dz"]
    for i in range(3):
        axes[i].plot(t, gt[:, i], label="GT", linewidth=1.2, color="black")
        axes[i].plot(t, pred[:, i], label="Pred", linewidth=1.0, color="red", alpha=0.8)
        axes[i].set_ylabel(f"{labels[i]} (m)", fontsize=11)
        axes[i].grid(linestyle=":", alpha=0.6)
        if i == 0:
            axes[i].legend(fontsize=9, loc="upper right")
    axes[-1].set_xlabel("Window Index", fontsize=11)
    fig.suptitle("Per-window dx/dy/dz", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_dp_body_series(gt_dp_body, pred_dp_body, vis_num, output_path):
    t = np.arange(vis_num)
    gt = gt_dp_body[:vis_num]
    pred = pred_dp_body[:vis_num]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=150, sharex=True)
    labels = ["dx", "dy", "dz"]
    for i in range(3):
        axes[i].plot(t, gt[:, i], label="GT", linewidth=1.2, color="black")
        axes[i].plot(t, pred[:, i], label="Pred", linewidth=1.0, color="royalblue", alpha=0.8)
        axes[i].set_ylabel(f"body {labels[i]} (m)", fontsize=11)
        axes[i].grid(linestyle=":", alpha=0.6)
        if i == 0:
            axes[i].legend(fontsize=9, loc="upper right")
    axes[-1].set_xlabel("Window Index", fontsize=11)
    fig.suptitle("Per-window body-frame dx/dy/dz", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_relative_euler(R_pred, R_gt, out_dir, seq_name, rotmat_to_euler_xyz):
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
        if i == 0:
            axes[i].legend(loc="upper right")

    axes[-1].set_xlabel("Window Index (Stride Steps)")
    fig.suptitle(f"Relative Rotation (Per Stride): {seq_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{seq_name}_1_rel.png"), dpi=100)
    plt.close(fig)


def plot_posenet_trajectory_comparison(init_pos, dl, yaw_pred_abs, yaw_gt_motion, out_dir, seq_name, reconstruct_fn):
    traj_pred = reconstruct_fn(init_pos, dl, yaw_pred_abs)
    traj_gt = reconstruct_fn(init_pos, dl, yaw_gt_motion)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(traj_gt[:, 0], traj_gt[:, 1], label="GT Trajectory", linewidth=2, color="black", alpha=0.6)
    ax.plot(traj_pred[:, 0], traj_pred[:, 1], label="Pred Yaw + GT Len", linewidth=1.5, color="orange", linestyle="--")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"Trajectory Reconstruction: {seq_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{seq_name}_2_traj.png"), dpi=100)
    plt.close(fig)


def plot_heading_analysis(yaw_phone_gt, yaw_phone_pred, heading_motion_gt, out_dir, seq_name, unwrap_fn):
    t = np.arange(len(yaw_phone_gt))
    deg = 180.0 / np.pi
    u_phone_gt = unwrap_fn(yaw_phone_gt)
    u_phone_pred = unwrap_fn(yaw_phone_pred)
    u_motion_gt = unwrap_fn(heading_motion_gt)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, u_motion_gt * deg, label="Motion Heading (GT)", color="#1f77b4", linestyle="--", linewidth=2)
    ax.plot(t, u_phone_gt * deg, label="Phone Yaw (GT)", color="#2ca02c", alpha=0.6, linewidth=2)
    ax.plot(t, u_phone_pred * deg, label="Phone Yaw (Pred Accumulated)", color="#ff7f0e", alpha=0.8, linewidth=1.5)
    ax.set_ylabel("Accumulated Yaw (deg)")
    ax.set_xlabel("Window Index")
    ax.set_title(f"Heading Analysis: {seq_name}\n(Drift Check)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{seq_name}_3_head.png"), dpi=100)
    plt.close(fig)


def plot_absolute_euler(R_pred, R_gt, out_dir, seq_name, rotmat_to_euler_xyz, wrap_angle):
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
    plt.close(fig)
