import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data.relative_window_truth import load_pose_imu_relative_datasets
from models.truth_autoencoder import TruthAutoEncoder


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
output_dir = os.path.join(PROJECT_DIR, "output", "disp_truth_autoencoder", dataset_name.lower())
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_torch_checkpoint(path, map_location):
    return torch.load(path, map_location=map_location, weights_only=False)


def normalize_truth(x, mean, std):
    return (x - mean) / std


def unnormalize_truth(x, mean, std):
    return x * std + mean


def make_loader(x_truth, x_imu):
    return DataLoader(
        TensorDataset(
            torch.tensor(x_truth, dtype=torch.float32),
            torch.tensor(x_imu, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def dense_rmse(pred, gt):
    err = pred - gt
    return float(np.sqrt(np.mean(np.sum(err * err, axis=2))))


def endpoint_rmse(pred, gt):
    err = pred[:, -1, :] - gt[:, -1, :]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def path_length(seq):
    steps = np.diff(seq, axis=1)
    return np.sum(np.linalg.norm(steps, axis=2), axis=1)


def run_inference(model, loader, mean_np, std_np):
    model.eval()
    losses = []
    pred_rows = []
    gt_rows = []
    with torch.no_grad():
        for xb, ib in loader:
            xb = xb.to(device)
            ib = ib.to(device)
            recon, latent = model(xb, imu_seq=ib)
            loss = F.smooth_l1_loss(recon, xb)
            losses.append(float(loss.item()) * xb.size(0))
            pred_rows.append(unnormalize_truth(recon.cpu().numpy(), mean_np, std_np))
            gt_rows.append(unnormalize_truth(xb.cpu().numpy(), mean_np, std_np))
            _ = latent
    pred = np.concatenate(pred_rows, axis=0)
    gt = np.concatenate(gt_rows, axis=0)
    metrics = {
        "loss": float(sum(losses) / max(len(gt), 1)),
        "dense_rmse": dense_rmse(pred, gt),
        "endpoint_rmse": endpoint_rmse(pred, gt),
        "path_len_mae": float(np.mean(np.abs(path_length(pred) - path_length(gt)))),
    }
    return pred, gt, metrics


def plot_examples(pred, gt, out_dir, max_examples=6):
    if pred.shape[0] == 0:
        return
    count = min(max_examples, pred.shape[0])
    idx = np.linspace(0, pred.shape[0] - 1, count, dtype=np.int64)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150)
    axes = axes.reshape(-1)
    for ax, sample_idx in zip(axes, idx):
        ax.plot(gt[sample_idx, :, 0], gt[sample_idx, :, 1], color="black", linewidth=1.6, label="GT")
        ax.plot(pred[sample_idx, :, 0], pred[sample_idx, :, 1], color="red", linewidth=1.2, alpha=0.85, label="Recon")
        ax.scatter(gt[sample_idx, 0, 0], gt[sample_idx, 0, 1], color="green", s=20)
        ax.scatter(gt[sample_idx, -1, 0], gt[sample_idx, -1, 1], color="purple", s=20)
        ax.axis("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(f"Val window {int(sample_idx)}")
    for ax in axes[count:]:
        ax.axis("off")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Relative displacement test reconstruction: {dataset_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_disp_reconstruction_examples.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    ckpt_path = os.path.join(ckpt_dir, "disp_truth_autoencoder.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = load_torch_checkpoint(ckpt_path, map_location=device)
    model = TruthAutoEncoder(**state["model_config"]).to(device)
    model.load_state_dict(state["model_state_dict"])

    mean_np = np.asarray(state["truth_mean"], dtype=np.float32).reshape(1, 1, -1)
    std_np = np.asarray(state["truth_std"], dtype=np.float32).reshape(1, 1, -1)
    datasets = load_pose_imu_relative_datasets(
        dataset_name,
        RIDI_ROOT,
        OXIOD_ROOT,
        window_size=int(state.get("window_size", window_size)),
        stride=int(state.get("stride", stride)),
        start_offset=0,
    )
    x_val = datasets["disp_val"]
    imu_val = datasets["imu_val"]
    if x_val.shape[0] == 0:
        raise RuntimeError(f"No displacement truth validation windows found for DATASET={dataset_name}")

    x_val_norm = normalize_truth(x_val, mean_np, std_np).astype(np.float32)
    pred, gt, metrics = run_inference(model, make_loader(x_val_norm, imu_val), mean_np, std_np)
    print(
        f"Val windows={x_val.shape[0]} loss={metrics['loss']:.6f} dense_rmse={metrics['dense_rmse']:.4f}m "
        f"endpoint_rmse={metrics['endpoint_rmse']:.4f}m path_len_mae={metrics['path_len_mae']:.4f}m"
    )
    plot_examples(pred, gt, output_dir)

    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "dataset": dataset_name,
                "num_val_windows": int(x_val.shape[0]),
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
