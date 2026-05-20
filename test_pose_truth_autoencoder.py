import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.relative_window_truth import load_pose_imu_relative_datasets
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
output_dir = os.path.join(PROJECT_DIR, "output", "pose_truth_autoencoder", dataset_name.lower())
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


def make_loader(x_truth, x_imu):
    return DataLoader(
        TensorDataset(
            torch.tensor(x_truth, dtype=torch.float32),
            torch.tensor(x_imu, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def evaluate(model, loader):
    model.eval()
    pred_rows = []
    gt_rows = []
    loss_rows = []
    end_rows = []
    with torch.no_grad():
        for xb, ib in loader:
            xb = xb.to(device)
            ib = ib.to(device)
            recon, latent = model(xb, imu_seq=ib)
            loss_rows.append(float(quaternion_sequence_loss(recon, xb).item()) * xb.size(0))
            end_rows.append(float(quaternion_endpoint_error_rad(recon, xb).item()) * xb.size(0))
            pred_rows.append(recon.cpu().numpy())
            gt_rows.append(xb.cpu().numpy())
            _ = latent
    pred = np.concatenate(pred_rows, axis=0)
    gt = np.concatenate(gt_rows, axis=0)
    yaw_pred = quat_to_yaw_np(pred)
    yaw_gt = quat_to_yaw_np(gt)
    yaw_err = angle_wrap(yaw_pred - yaw_gt)
    metrics = {
        "loss": float(sum(loss_rows) / max(len(gt), 1)),
        "yaw_rmse_deg": float(np.sqrt(np.mean(yaw_err * yaw_err)) * 180.0 / np.pi),
        "endpoint_angle_deg": float(sum(end_rows) / max(len(gt), 1) * 180.0 / np.pi),
    }
    return pred, gt, metrics


def run_model_numpy(model, x, imu):
    xb = torch.tensor(x, dtype=torch.float32, device=device)
    ib = torch.tensor(imu, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon, _latent = model(xb, imu_seq=ib)
    return recon.cpu().numpy()


def sequence_metrics(pred, gt):
    yaw_pred = quat_to_yaw_np(pred)
    yaw_gt = quat_to_yaw_np(gt)
    yaw_err = angle_wrap(yaw_pred - yaw_gt)
    yaw_rmse_deg = float(np.sqrt(np.mean(yaw_err * yaw_err)) * 180.0 / np.pi)
    q_pred = torch.tensor(pred, dtype=torch.float32)
    q_gt = torch.tensor(gt, dtype=torch.float32)
    endpoint_angle_deg = float(quaternion_endpoint_error_rad(q_pred, q_gt).item() * 180.0 / np.pi)
    loss = float(quaternion_sequence_loss(q_pred, q_gt).item())
    return {
        "loss": loss,
        "yaw_rmse_deg": yaw_rmse_deg,
        "endpoint_angle_deg": endpoint_angle_deg,
    }


def plot_sequence_reconstruction(seq_name, pred, gt, out_dir):
    t = np.arange(pred.shape[0])
    yaw_pred = quat_to_yaw_np(pred)
    yaw_gt = quat_to_yaw_np(gt)
    if yaw_pred.ndim == 2:
        yaw_pred_vis = yaw_pred[:, -1]
        yaw_gt_vis = yaw_gt[:, -1]
        x_label = "Window Index In Sequence"
        title_suffix = "window-end relative yaw"
    else:
        yaw_pred_vis = yaw_pred
        yaw_gt_vis = yaw_gt
        x_label = "Frame Index"
        title_suffix = "relative yaw"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)
    axes[0].plot(t, yaw_gt_vis * 180.0 / np.pi, color="black", linewidth=1.4)
    axes[0].plot(t, yaw_pred_vis * 180.0 / np.pi, color="red", linewidth=1.0, alpha=0.85)
    axes[0].set_ylabel("Yaw (deg)")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].text(
        0.01,
        0.98,
        "black: GT yaw\nred: Recon yaw",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    yaw_err_deg = angle_wrap(yaw_pred_vis - yaw_gt_vis) * 180.0 / np.pi
    axes[1].plot(t, yaw_err_deg, color="royalblue", linewidth=1.0)
    axes[1].set_ylabel("Yaw Error (deg)")
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(f"Relative pose reconstruction: {seq_name} ({title_suffix})")
    plt.tight_layout()
    safe_name = seq_name.replace("/", "_").replace("\\", "_")
    plt.savefig(os.path.join(out_dir, f"{safe_name}_yaw.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    ckpt_path = os.path.join(ckpt_dir, "pose_truth_autoencoder.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = load_torch_checkpoint(ckpt_path, map_location=device)
    model = PoseTruthAutoEncoder(**state["model_config"]).to(device)
    model.load_state_dict(state["model_state_dict"])

    datasets = load_pose_imu_relative_datasets(
        dataset_name,
        RIDI_ROOT,
        OXIOD_ROOT,
        window_size=int(state.get("window_size", window_size)),
        stride=int(state.get("stride", stride)),
        start_offset=0,
    )
    x_val = datasets["pose_val"]
    imu_val = datasets["imu_val"]
    val_sequences = datasets["val_sequences"]
    if x_val.shape[0] == 0:
        raise RuntimeError(f"No pose truth validation windows found for DATASET={dataset_name}")

    pred, gt, metrics = evaluate(model, make_loader(x_val, imu_val))
    print(
        f"Val windows={x_val.shape[0]} loss={metrics['loss']:.6f} "
        f"yaw_rmse={metrics['yaw_rmse_deg']:.3f}deg endpoint_angle={metrics['endpoint_angle_deg']:.3f}deg"
    )

    sequence_rows = []
    for seq in val_sequences:
        seq_name = seq["name"]
        gt_seq = seq["truth"]
        pred_seq = run_model_numpy(model, gt_seq, seq["imu"])
        row = {"name": seq_name, **sequence_metrics(pred_seq, gt_seq)}
        sequence_rows.append(row)
        print(
            f"[{seq_name}] loss={row['loss']:.6f} yaw_rmse={row['yaw_rmse_deg']:.3f}deg "
            f"endpoint_angle={row['endpoint_angle_deg']:.3f}deg"
        )
        plot_sequence_reconstruction(seq_name, pred_seq, gt_seq, output_dir)

    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
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
