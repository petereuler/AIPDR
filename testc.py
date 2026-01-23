import os
import numpy as np
import torch

from data.dataset_RIDI import load_ridi_raw, window_dataset as ridi_window
from models.pose_net import PoseNet, quat_to_rotmat, rotate_imu
from models.navigator import Navigator
from utils.navigator_pipeline import accumulate_rotations, compute_init_rot
from utils.visualization import plot_trajectory_comparison


window_size = 320
stride = 64
batch_size = 256
use_gt_pose = True

project_dir = "/home/admin407/code/zyshe/NavCorrector"
ridi_root = os.path.join(project_dir, "RIDI")
ckpt_dir = os.path.join(project_dir, "checkpoints_cls")
output_dir = os.path.join(project_dir, "output", "navigator_viz")
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def predict_navigator_batches(pose_net, navigator, gx, ax, seq_id, init_rot, use_gt_pose=False, yori=None):
    n = gx.shape[0]
    preds_len = []
    preds_head = []
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
                xb_global = rotate_imu(xb, R_abs)
            else:
                sid = seq_id[start:end]
                irot = init_rot[start:end]
                R_delta = pose_net(xb)
                R_abs = accumulate_rotations(R_delta, sid, irot)
                xb_global = rotate_imu(xb, R_abs)
            pred_len, pred_h, _, pred_dp = navigator(xb_global)
            preds_len.append(pred_len.cpu().numpy())
            preds_head.append(pred_h.cpu().numpy())
            preds_dp.append(pred_dp.cpu().numpy())

    pred_len = np.concatenate(preds_len, axis=0)
    pred_head = np.concatenate(preds_head, axis=0)
    pred_dp = np.concatenate(preds_dp, axis=0)
    return pred_len, pred_head, pred_dp


def build_traj_from_delta_p(init_pos, dp):
    traj = np.zeros((dp.shape[0], 2), dtype=np.float32)
    traj[:, 0] = init_pos[0] + np.cumsum(dp[:, 0])
    traj[:, 1] = init_pos[1] + np.cumsum(dp[:, 1])
    return traj


def main():
    pose_ckpt = os.path.join(ckpt_dir, "pose_net.pth")
    nav_ckpt = os.path.join(ckpt_dir, "navigator.pth")

    pose_net = None
    navigator = Navigator(imu_dim=6, feat_dim=64).to(device)

    if not use_gt_pose:
        pose_net = PoseNet(imu_dim=6, hidden_dim=128).to(device)
        pose_net.load_state_dict(torch.load(pose_ckpt, map_location=device))
    navigator.load_state_dict(torch.load(nav_ckpt, map_location=device))

    test_list = os.path.join(ridi_root, "data", "list_test_publish_v2.txt")
    with open(test_list, "r") as f:
        seq_names = [line.strip().split(",")[0] for line in f if line.strip()]

    for name in seq_names:
        seq_dir = os.path.join(ridi_root, "data", name)
        if not os.path.isdir(seq_dir):
            continue
        gyro, acc, pos3d, ori = load_ridi_raw(seq_dir)

        [gx, ax], [dl, dh, yabs, yori, yrel, ydp], init_pos, _ = ridi_window(
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

        pred_len, pred_head, pred_dp = predict_navigator_batches(
            pose_net, navigator, gx, ax, seq_id, init_rot, use_gt_pose=use_gt_pose, yori=yori
        )

        gt_traj = build_traj_from_delta_p(init_pos, ydp[:, :2])
        pred_traj = build_traj_from_delta_p(init_pos, pred_dp[:, :2])

        gt_len = dl.reshape(-1)
        pred_len = pred_len.reshape(-1)
        len_mae = np.abs(pred_len - gt_len).mean()

        gt_head = np.arctan2(ydp[:, 1], ydp[:, 0])
        pred_head = pred_head.reshape(-1)
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


if __name__ == "__main__":
    main()
