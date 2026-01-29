import os
import numpy as np
import torch

from data.dataset_RIDI import load_ridi_raw, window_dataset as ridi_window
from models.pose_net import PoseNetTransformer, quat_conj, quat_to_rotmat, rotate_imu
from models.navigator import Navigator
from utils.navigator_pipeline import accumulate_rotations, compute_init_rot
from utils.visualization import plot_trajectory_comparison, plot_time_series, plot_cumulative_series
from matplotlib import pyplot as plt

window_size = 320
stride = 64
batch_size = 256
use_gt_pose = False
pose_output_mode = "6d"

project_dir = "/home/admin407/code/zyshe/NavCorrector"
ridi_root = os.path.join(project_dir, "RIDI")
ckpt_dir = os.path.join(project_dir, "checkpoints_cls")
output_dir = os.path.join(project_dir, "output", "navigator_viz")
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def predict_navigator_batches(pose_net, navigator, gx, ax, seq_id, init_rot, use_gt_pose=False, yori=None, R_align=None):
    n = gx.shape[0]
    preds_dp = []

    if not use_gt_pose:
        pose_net.eval()
    navigator.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.tensor(np.concatenate([gx[start:end], ax[start:end]], axis=-1),
                              dtype=torch.float32, device=device)
            if use_gt_pose:
                if yori is None:
                    raise ValueError("use_gt_pose=True requires yori.")
                yb_ori = torch.tensor(yori[start:end], dtype=torch.float32, device=device)
                R_abs = quat_to_rotmat(yb_ori)
            else:
                sid = seq_id[start:end]
                irot = init_rot[start:end]
                R_delta = pose_net(xb)
                R_abs = accumulate_rotations(R_delta, sid, irot)
            R_use = R_abs
            if R_align is not None:
                R_use = torch.matmul(R_align[start:end], R_abs)
            xb_global = rotate_imu(xb, R_use)
            pred_dp = navigator(xb_global)
            if R_align is not None:
                pred_dp = torch.matmul(R_align[start:end].transpose(1, 2), pred_dp.unsqueeze(-1)).squeeze(-1)
            preds_dp.append(pred_dp.cpu().numpy())

    pred_dp = np.concatenate(preds_dp, axis=0)
    return pred_dp


def build_traj_from_delta_p(init_pos, dp):
    traj = np.zeros((dp.shape[0], 2), dtype=np.float32)
    traj[:, 0] = init_pos[0] + np.cumsum(dp[:, 0])
    traj[:, 1] = init_pos[1] + np.cumsum(dp[:, 1])
    return traj


def plot_absolute_heading(gt_head, pred_head, vis_num, output_path):
    t = np.arange(vis_num)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=150)
    ax.plot(t, np.degrees(gt_head[:vis_num]), label="GT Heading", linewidth=1.2, color="black")
    ax.plot(t, np.degrees(pred_head[:vis_num]), label="Pred Heading", linewidth=1.0, color="red", alpha=0.8)
    ax.set_title("Absolute Heading Over Time", fontsize=14)
    ax.set_ylabel("Heading (deg)", fontsize=12)
    ax.set_xlabel("Window Index", fontsize=12)
    ax.grid(linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
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


def main():
    pose_ckpt = os.path.join(ckpt_dir, "pose_net.pth")
    nav_ckpt = os.path.join(ckpt_dir, "navigator.pth")

    navigator = Navigator(imu_dim=6, feat_dim=64).to(device)

    pose_net = None
    if not use_gt_pose:
        pose_net = PoseNetTransformer(
            imu_dim=6,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            output_mode=pose_output_mode,
        ).to(device)
        pose_net.load_state_dict(torch.load(pose_ckpt, map_location=device))
    navigator.load_state_dict(torch.load(nav_ckpt, map_location=device))

    test_list = os.path.join(ridi_root, "data", "list_test_publish_v2.txt")
    with open(test_list, "r") as f:
        seq_names = [line.strip().split(",")[0] for line in f if line.strip()]

    for name in seq_names:
        seq_dir = os.path.join(ridi_root, "data", name)
        if not os.path.isdir(seq_dir):
            continue
        gyro, acc, pos3d, ori = load_ridi_raw(seq_dir, acc_source="linacce")

        [gx, ax], [dl, dh, _yabs, yori, _yrel, ydp], init_pos, init_head = ridi_window(
            gyro, acc, pos3d, ori,
            mode="2d",
            window_size=window_size,
            stride=stride,
            filter_window=0,
            smooth_heading=False,
            smooth_length=False,
            return_abs_heading=True,
            return_ori=True,
            return_rel_ori=True,
            return_delta_p=True,
        )
        if gx.shape[0] == 0:
            continue

        if use_gt_pose:
            init_rot = None
            seq_id = None
        else:
            init_rot_np = compute_init_rot(ori, pos3d, window_size, stride)
            init_rot = torch.tensor(init_rot_np, dtype=torch.float32, device=device)
            seq_id = torch.zeros(gx.shape[0], dtype=torch.int64, device=device)

        q0 = torch.tensor(ori[0].astype(np.float32), device=device)
        q_align = quat_conj(q0.unsqueeze(0))
        R_align = quat_to_rotmat(q_align).repeat(gx.shape[0], 1, 1)

        pred_dp = predict_navigator_batches(
            pose_net, navigator, gx, ax, seq_id, init_rot, use_gt_pose=use_gt_pose, yori=yori, R_align=R_align
        )

        gt_traj = build_traj_from_delta_p(init_pos, ydp[:, :2])
        pred_traj = build_traj_from_delta_p(init_pos, pred_dp[:, :2])

        gt_len = np.linalg.norm(ydp[:, :2], axis=1)
        pred_len = np.linalg.norm(pred_dp[:, :2], axis=1)
        len_mae = np.abs(pred_len - gt_len).mean()

        gt_head = np.arctan2(ydp[:, 1], ydp[:, 0])
        pred_head = np.arctan2(pred_dp[:, 1], pred_dp[:, 0])
        diff = (pred_head - gt_head + np.pi) % (2 * np.pi) - np.pi
        head_mae = np.abs(diff).mean()

        traj_err = np.linalg.norm(pred_traj - gt_traj, axis=1)
        traj_rmse = np.sqrt(np.mean(traj_err ** 2))

        print(f"[{name}] Len MAE: {len_mae:.4f} | Head MAE: {np.degrees(head_mae):.2f} deg | Traj RMSE: {traj_rmse:.3f} m")

        plot_trajectory_comparison(
            gt_traj,
            gt_traj,
            pred_traj,
            output_dir,
            name,
        )

        pred_len_plot = pred_len.reshape(-1, 1)
        pred_head_plot = pred_head.reshape(-1, 1)
        vis_num = min(1000, len(dl))
        gt_len_plot = gt_len.reshape(-1, 1)
        gt_head_plot = gt_head.reshape(-1, 1)
        plot_time_series(
            gt_len_plot,
            gt_head_plot,
            pred_len_plot,
            pred_head_plot,
            vis_num,
            os.path.join(output_dir, f"{name}_heading_timeseries.png"),
        )
        plot_absolute_heading(
            gt_head,
            pred_head,
            vis_num,
            os.path.join(output_dir, f"{name}_heading_cumulative.png"),
        )

        plot_cumulative_dp(
            ydp,
            pred_dp,
            vis_num,
            os.path.join(output_dir, f"{name}_dp_cumulative.png"),
        )


if __name__ == "__main__":
    main()
