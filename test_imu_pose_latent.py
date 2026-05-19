import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.relative_window_truth import load_pose_imu_relative_datasets
from models.imu_to_pose_latent import IMUToPoseLatent
from models.pose_truth_autoencoder import (
    PoseTruthAutoEncoder,
    quaternion_endpoint_error_rad,
    quaternion_sequence_loss,
)


WINDOW_SIZE = 64
STRIDE = 64
DATASET = "RIDI"
BATCH_SIZE = 512

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RIDI_ROOT = os.path.join(PROJECT_DIR, "RIDI")
OXIOD_ROOT = os.path.join(PROJECT_DIR, "OXIOD")

dataset_name = os.getenv("DATASET", DATASET).upper()
window_size = int(os.getenv("WINDOW_SIZE", WINDOW_SIZE))
stride = int(os.getenv("STRIDE", STRIDE))
batch_size = int(os.getenv("BATCH_SIZE", BATCH_SIZE))

ckpt_dir = os.path.join(PROJECT_DIR, "checkpoints", dataset_name.lower())
output_dir = os.path.join(PROJECT_DIR, "output", "imu_pose_latent", dataset_name.lower())
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_torch_checkpoint(path, map_location):
    return torch.load(path, map_location=map_location, weights_only=False)


def quat_to_yaw_np(q):
    q = np.asarray(q, dtype=np.float32)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def angle_wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def quat_to_rotmat_np(q):
    q = np.asarray(q, dtype=np.float32)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32)


def make_loader(x_imu, y_pose):
    ds = TensorDataset(
        torch.tensor(x_imu, dtype=torch.float32),
        torch.tensor(y_pose, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def evaluate(predictor, autoencoder, loader):
    predictor.eval()
    autoencoder.eval()
    pred_rows = []
    gt_rows = []
    loss_rows = []
    end_rows = []
    with torch.no_grad():
        for xb_imu, yb_pose in loader:
            xb_imu = xb_imu.to(device)
            yb_pose = yb_pose.to(device)
            z_pred = predictor(xb_imu)
            q_pred = autoencoder.decode(z_pred)
            loss_rows.append(float(quaternion_sequence_loss(q_pred, yb_pose).item()) * xb_imu.size(0))
            end_rows.append(float(quaternion_endpoint_error_rad(q_pred, yb_pose).item()) * xb_imu.size(0))
            pred_rows.append(q_pred.cpu().numpy())
            gt_rows.append(yb_pose.cpu().numpy())
    pred = np.concatenate(pred_rows, axis=0)
    gt = np.concatenate(gt_rows, axis=0)
    yaw_pred = quat_to_yaw_np(pred[:, -1, :])
    yaw_gt = quat_to_yaw_np(gt[:, -1, :])
    yaw_err = angle_wrap(yaw_pred - yaw_gt)
    metrics = {
        "loss": float(sum(loss_rows) / max(len(gt), 1)),
        "yaw_rmse_deg": float(np.sqrt(np.mean(yaw_err * yaw_err)) * 180.0 / np.pi),
        "endpoint_angle_deg": float(sum(end_rows) / max(len(gt), 1) * 180.0 / np.pi),
    }
    return pred, gt, metrics


def run_sequence(model, autoencoder, imu_seq):
    xb = torch.tensor(imu_seq, dtype=torch.float32, device=device)
    with torch.no_grad():
        z_pred = model(xb)
        q_pred = autoencoder.decode(z_pred)
    return q_pred.cpu().numpy()


def sequence_metrics(pred, gt):
    yaw_pred = quat_to_yaw_np(pred[:, -1, :])
    yaw_gt = quat_to_yaw_np(gt[:, -1, :])
    yaw_err = angle_wrap(yaw_pred - yaw_gt)
    q_pred = torch.tensor(pred, dtype=torch.float32)
    q_gt = torch.tensor(gt, dtype=torch.float32)
    return {
        "loss": float(quaternion_sequence_loss(q_pred, q_gt).item()),
        "yaw_rmse_deg": float(np.sqrt(np.mean(yaw_err * yaw_err)) * 180.0 / np.pi),
        "endpoint_angle_deg": float(quaternion_endpoint_error_rad(q_pred, q_gt).item() * 180.0 / np.pi),
    }


def rollout_world_trajectory_from_step_disp_and_pose(step_disp_body_seq, pose_seq, init_rot):
    current_r = np.asarray(init_rot, dtype=np.float32).copy()
    pos_xy = np.zeros(2, dtype=np.float32)
    traj_rows = []
    for win_idx in range(len(step_disp_body_seq)):
        rel_rot_seq = quat_to_rotmat_np(pose_seq[win_idx])
        for step_idx in range(1, step_disp_body_seq.shape[1]):
            r_step = current_r @ rel_rot_seq[step_idx - 1]
            dp_world = r_step @ step_disp_body_seq[win_idx, step_idx]
            pos_xy = pos_xy + dp_world[:2]
        traj_rows.append(pos_xy.copy())
        current_r = current_r @ rel_rot_seq[-1]
    return np.asarray(traj_rows, dtype=np.float32)


def trajectory_metrics(pred_traj, gt_traj):
    err = pred_traj - gt_traj
    traj_rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
    final_err = float(np.linalg.norm(err[-1])) if len(err) > 0 else 0.0
    return {"traj_rmse": traj_rmse, "final_err": final_err}


def plot_sequence(seq_name, pred, gt, out_dir):
    t = np.arange(pred.shape[0])
    yaw_pred = quat_to_yaw_np(pred[:, -1, :])
    yaw_gt = quat_to_yaw_np(gt[:, -1, :])
    yaw_err_deg = angle_wrap(yaw_pred - yaw_gt) * 180.0 / np.pi

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)
    axes[0].plot(t, yaw_gt * 180.0 / np.pi, color="black", linewidth=1.4)
    axes[0].plot(t, yaw_pred * 180.0 / np.pi, color="red", linewidth=1.0, alpha=0.85)
    axes[0].set_ylabel("Window-End Relative Yaw (deg)")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].text(
        0.01,
        0.98,
        "black: GT yaw\nred: Pred yaw",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    axes[1].plot(t, yaw_err_deg, color="royalblue", linewidth=1.0)
    axes[1].set_ylabel("Yaw Error (deg)")
    axes[1].set_xlabel("Window Index In Sequence")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(f"IMU to pose latent: {seq_name}")
    plt.tight_layout()
    safe_name = seq_name.replace("/", "_").replace("\\", "_")
    plt.savefig(os.path.join(out_dir, f"{safe_name}_yaw.png"), bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(seq_name, pred_traj, gt_traj, out_dir):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], color="black", linewidth=1.6, label="GT traj")
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], color="red", linewidth=1.2, alpha=0.9, label="Pred pose + GT step disp")
    ax.scatter(gt_traj[0, 0], gt_traj[0, 1], color="green", s=30)
    ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], color="purple", s=30)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"Trajectory from predicted pose: {seq_name}")
    plt.tight_layout()
    safe_name = seq_name.replace("/", "_").replace("\\", "_")
    plt.savefig(os.path.join(out_dir, f"{safe_name}_traj.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    pose_ae_ckpt = os.path.join(ckpt_dir, "pose_truth_autoencoder.pth")
    predictor_ckpt = os.path.join(ckpt_dir, "imu_to_pose_latent.pth")
    if not os.path.exists(pose_ae_ckpt):
        raise FileNotFoundError(f"Pose autoencoder checkpoint not found: {pose_ae_ckpt}")
    if not os.path.exists(predictor_ckpt):
        raise FileNotFoundError(f"IMU->pose latent checkpoint not found: {predictor_ckpt}")

    pose_ae_state = load_torch_checkpoint(pose_ae_ckpt, map_location=device)
    predictor_state = load_torch_checkpoint(predictor_ckpt, map_location=device)

    autoencoder = PoseTruthAutoEncoder(**pose_ae_state["model_config"]).to(device)
    autoencoder.load_state_dict(pose_ae_state["model_state_dict"])
    predictor = IMUToPoseLatent(**predictor_state["model_config"]).to(device)
    predictor.load_state_dict(predictor_state["model_state_dict"])

    datasets = load_pose_imu_relative_datasets(
        dataset_name,
        RIDI_ROOT,
        OXIOD_ROOT,
        window_size=int(predictor_state.get("window_size", window_size)),
        stride=int(predictor_state.get("stride", stride)),
        start_offset=0,
    )
    x_val = datasets["imu_val"]
    y_val = datasets["pose_val"]
    val_sequences = datasets["val_sequences"]
    if len(x_val) == 0:
        raise RuntimeError(f"No aligned IMU-pose validation windows found for DATASET={dataset_name}")

    pred, gt, metrics = evaluate(predictor, autoencoder, make_loader(x_val, y_val))
    print(
        f"Val windows={x_val.shape[0]} loss={metrics['loss']:.6f} "
        f"yaw_rmse={metrics['yaw_rmse_deg']:.3f}deg endpoint_angle={metrics['endpoint_angle_deg']:.3f}deg"
    )

    sequence_rows = []
    for seq in val_sequences:
        seq_name = seq["name"]
        pred_seq = run_sequence(predictor, autoencoder, seq["imu"])
        pose_row = sequence_metrics(pred_seq, seq["truth"])
        gt_traj = seq["gt_traj"]
        pred_traj = rollout_world_trajectory_from_step_disp_and_pose(seq["step_disp"], pred_seq, seq["init_rot"])
        traj_row = trajectory_metrics(pred_traj, gt_traj)
        row = {
            "name": seq_name,
            **pose_row,
            "traj_rmse": traj_row["traj_rmse"],
            "final_err": traj_row["final_err"],
        }
        sequence_rows.append(row)
        print(
            f"[{seq_name}] loss={row['loss']:.6f} yaw_rmse={row['yaw_rmse_deg']:.3f}deg "
            f"endpoint_angle={row['endpoint_angle_deg']:.3f}deg "
            f"traj_rmse={row['traj_rmse']:.3f}m final_err={row['final_err']:.3f}m"
        )
        plot_sequence(seq_name, pred_seq, seq["truth"], output_dir)
        plot_trajectory(seq_name, pred_traj, gt_traj, output_dir)

    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "checkpoint": predictor_ckpt,
                "pose_autoencoder_checkpoint": pose_ae_ckpt,
                "dataset": dataset_name,
                "num_val_windows": int(x_val.shape[0]),
                "metrics": metrics,
                "sequence_rows": sequence_rows,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
