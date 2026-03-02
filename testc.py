import os
import numpy as np
import torch

from data.dataset_RIDI import load_ridi_raw, window_dataset as ridi_window
from models.pose_net import PoseNetTransformer, quat_to_rotmat
from models.navigator import Navigator
from utils.navigator_pipeline import compute_init_rot
from utils.visualization import plot_trajectory_comparison
from matplotlib import pyplot as plt

window_size = 64
stride = 64
batch_size = 256
use_gt_pose = False

project_dir = "/home/admin407/code/zyshe/NavCorrector"
ridi_root = os.path.join(project_dir, "RIDI")
ckpt_dir = os.path.join(project_dir, "checkpoints_cls")
output_dir = os.path.join(project_dir, "output", "navigator_viz")
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



def predict_navigator_batches(
    pose_net,
    navigator,
    gx_pose,
    ax_pose,
    gx_nav,
    ax_nav,
    init_rot,
    use_gt_pose=False,
    yori=None,
    yrel=None,
    R_align=None,
):
    n = gx_nav.shape[0]
    preds_dp = []
    current_R_end = init_rot

    if not use_gt_pose:
        pose_net.eval()
    navigator.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb_pose = torch.tensor(
                np.concatenate([gx_pose[start:end], ax_pose[start:end]], axis=-1),
                dtype=torch.float32,
                device=device,
            )
            xb_nav = torch.tensor(
                np.concatenate([gx_nav[start:end], ax_nav[start:end]], axis=-1),
                dtype=torch.float32,
                device=device,
            )
            if use_gt_pose:
                if yori is None:
                    raise ValueError("use_gt_pose=True requires yori.")
                if yrel is None:
                    raise ValueError("use_gt_pose=True requires yrel.")
                yb_ori = torch.tensor(yori[start:end], dtype=torch.float32, device=device)
                yb_rel = torch.tensor(yrel[start:end], dtype=torch.float32, device=device)
                R_end = quat_to_rotmat(yb_ori)
                R_rel = quat_to_rotmat(yb_rel)
                R_start = torch.matmul(R_end, R_rel.transpose(1, 2))
                R_use = R_start
            else:
                q_rel = pose_net(xb_pose)
                R_delta = quat_to_rotmat(q_rel)
                R_start_list = []
                for i in range(R_delta.size(0)):
                    R_start_i = current_R_end
                    R_start_list.append(R_start_i)
                    current_R_end = torch.matmul(current_R_end, R_delta[i])
                R_use = torch.stack(R_start_list, dim=0)
            if R_align is not None:
                R_use = torch.matmul(R_align[start:end], R_use)
            pred_dp_body = navigator(xb_nav)
            pred_dp = torch.matmul(R_use, pred_dp_body.unsqueeze(-1)).squeeze(-1)
            preds_dp.append(pred_dp.cpu().numpy())

    pred_dp = np.concatenate(preds_dp, axis=0)
    return pred_dp


def build_traj_from_delta_p(init_pos, dp):
    traj = np.zeros((dp.shape[0], 2), dtype=np.float32)
    traj[:, 0] = init_pos[0] + np.cumsum(dp[:, 0])
    traj[:, 1] = init_pos[1] + np.cumsum(dp[:, 1])
    return traj


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
        gyro_pose, acc_pose, pos_xyz, ori = load_ridi_raw(seq_dir, acc_source="acce")
        gyro_nav, acc_nav, _pos_nav, _ori_nav = load_ridi_raw(seq_dir, acc_source="linacce")

        [gx_nav, ax_nav], [dl, yori, _yrel, ydp, ydp_world], init_pos, init_head = ridi_window(
            gyro_nav, acc_nav, pos_xyz, ori,
            window_size=window_size,
            stride=stride,
            filter_window=0,
            smooth_length=False,
            return_ori=True,
            return_rel_ori=True,
            return_delta_p=True,
            return_delta_p_world=True,
        )
        [gx_pose, ax_pose], _labels_pose, _init_pos_pose, _init_head_pose = ridi_window(
            gyro_pose, acc_pose, pos_xyz, ori,
            window_size=window_size,
            stride=stride,
            filter_window=0,
            smooth_length=False,
            return_ori=False,
            return_rel_ori=False,
            return_delta_p=False,
            return_delta_p_world=False,
        )
        if gx_nav.shape[0] == 0 or gx_pose.shape[0] == 0:
            continue

        if use_gt_pose:
            init_rot = torch.eye(3, dtype=torch.float32, device=device)
        else:
            init_rot_np = compute_init_rot(ori, pos_xyz, window_size, stride)
            if init_rot_np.shape[0] == 0:
                continue
            init_rot = torch.tensor(init_rot_np[0], dtype=torch.float32, device=device)

        m = min(
            gx_pose.shape[0],
            gx_nav.shape[0],
            len(dl),
            len(yori),
            len(_yrel),
            len(ydp),
            len(ydp_world),
        )
        gx_pose = gx_pose[:m]
        ax_pose = ax_pose[:m]
        gx_nav = gx_nav[:m]
        ax_nav = ax_nav[:m]
        dl = dl[:m]
        yori = yori[:m]
        _yrel = _yrel[:m]
        ydp = ydp[:m]
        ydp_world = ydp_world[:m]

        pred_dp = predict_navigator_batches(
            pose_net,
            navigator,
            gx_pose,
            ax_pose,
            gx_nav,
            ax_nav,
            init_rot,
            use_gt_pose=use_gt_pose,
            yori=yori,
            yrel=_yrel,
            R_align=None,
        )


        gt_traj = build_traj_from_delta_p(init_pos, ydp_world[:, :2])
        pred_traj = build_traj_from_delta_p(init_pos, pred_dp[:, :2])

        gt_len = np.linalg.norm(ydp_world[:, :2], axis=1)
        pred_len = np.linalg.norm(pred_dp[:, :2], axis=1)
        len_mae = np.abs(pred_len - gt_len).mean()

        traj_err = np.linalg.norm(pred_traj - gt_traj, axis=1)
        traj_rmse = np.sqrt(np.mean(traj_err ** 2))

        print(f"[{name}] Len MAE: {len_mae:.4f} | Traj RMSE: {traj_rmse:.3f} m")

        plot_trajectory_comparison(
            gt_traj,
            gt_traj,
            pred_traj,
            output_dir,
            name,
        )

        vis_num = min(1000, len(dl))
        plot_cumulative_dp(
            ydp_world,
            pred_dp,
            vis_num,
            os.path.join(output_dir, f"{name}_dp_cumulative.png"),
        )


if __name__ == "__main__":
    main()
