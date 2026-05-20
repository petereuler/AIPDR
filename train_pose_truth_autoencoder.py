import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
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
EPOCHS = 1200
LR = 2e-4
WEIGHT_DECAY = 1e-4
LATENT_DIM = 64
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
IMU_CONDITION_MODE = "both"  # "none" | "encoder" | "decoder" | "both"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RIDI_ROOT = os.path.join(PROJECT_DIR, "RIDI")
OXIOD_ROOT = os.path.join(PROJECT_DIR, "OXIOD")

dataset_name = os.getenv("DATASET", DATASET).upper()
window_size = int(os.getenv("WINDOW_SIZE", WINDOW_SIZE))
stride = int(os.getenv("STRIDE", STRIDE))
batch_size = int(os.getenv("BATCH_SIZE", BATCH_SIZE))
epochs = int(os.getenv("EPOCHS", EPOCHS))
latent_dim = int(os.getenv("LATENT_DIM", LATENT_DIM))
mode = os.getenv("MODE", "resume").lower()
imu_condition_mode = os.getenv("IMU_CONDITION_MODE", IMU_CONDITION_MODE).lower()

ckpt_dir = os.path.join(PROJECT_DIR, "checkpoints", dataset_name.lower())
output_dir = os.path.join(PROJECT_DIR, "output", "pose_truth_autoencoder", dataset_name.lower())
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_torch_checkpoint(path, map_location):
    return torch.load(path, map_location=map_location, weights_only=False)


def validate_mode(mode):
    if mode not in {"resume", "retrain"}:
        raise ValueError(f"MODE must be 'resume' or 'retrain', got {mode}")


def default_model_config():
    return {
        "seq_len": window_size,
        "latent_dim": latent_dim,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "condition_mode": imu_condition_mode,
        "imu_dim": 6,
    }


def make_loader(x_truth, x_imu, shuffle):
    return DataLoader(
        TensorDataset(
            torch.tensor(x_truth, dtype=torch.float32),
            torch.tensor(x_imu, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def quat_to_yaw_np(q):
    q = np.asarray(q, dtype=np.float32)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def angle_wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def evaluate(model, loader):
    model.eval()
    loss_rows = []
    yaw_rmse_rows = []
    end_err_rows = []
    with torch.no_grad():
        for xb, ib in loader:
            xb = xb.to(device)
            ib = ib.to(device)
            recon, latent = model(xb, imu_seq=ib)
            loss_recon = quaternion_sequence_loss(recon, xb)
            end_err = quaternion_endpoint_error_rad(recon, xb)
            loss_rows.append(float(loss_recon.item()) * xb.size(0))
            end_err_rows.append(float(end_err.item()) * xb.size(0))

            yaw_pred = quat_to_yaw_np(recon.cpu().numpy())
            yaw_gt = quat_to_yaw_np(xb.cpu().numpy())
            yaw_err = angle_wrap(yaw_pred - yaw_gt)
            yaw_rmse = np.sqrt(np.mean(yaw_err * yaw_err))
            yaw_rmse_rows.append(float(yaw_rmse) * xb.size(0))
            _ = latent
    num = len(loader.dataset)
    return {
        "loss": float(sum(loss_rows) / max(num, 1)),
        "yaw_rmse_deg": float(sum(yaw_rmse_rows) / max(num, 1) * 180.0 / np.pi),
        "endpoint_angle_deg": float(sum(end_err_rows) / max(num, 1) * 180.0 / np.pi),
    }


def plot_pose_examples(model, x_val, imu_val, out_dir, max_examples=6):
    if x_val.shape[0] == 0:
        return
    count = min(max_examples, x_val.shape[0])
    idx = np.linspace(0, x_val.shape[0] - 1, count, dtype=np.int64)
    xb = torch.tensor(x_val[idx], dtype=torch.float32, device=device)
    ib = torch.tensor(imu_val[idx], dtype=torch.float32, device=device)
    with torch.no_grad():
        recon, _latent = model(xb, imu_seq=ib)
    pred = recon.cpu().numpy()
    gt = xb.cpu().numpy()
    yaw_pred = quat_to_yaw_np(pred)
    yaw_gt = quat_to_yaw_np(gt)
    t = np.arange(gt.shape[1])
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150, sharex=True)
    axes = axes.reshape(-1)
    for ax, i in zip(axes, range(count)):
        ax.plot(t, yaw_gt[i] * 180.0 / np.pi, color="black", linewidth=1.5, label="GT yaw")
        ax.plot(t, yaw_pred[i] * 180.0 / np.pi, color="red", linewidth=1.1, alpha=0.85, label="Recon yaw")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(f"Val window {int(idx[i])}")
    for ax in axes[count:]:
        ax.axis("off")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Relative pose reconstruction: {dataset_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pose_reconstruction_examples.png"), bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(path, model, val_metrics):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": default_model_config(),
            "window_size": window_size,
            "stride": stride,
            "dataset": dataset_name,
            "val_metrics": val_metrics,
        },
        path,
    )


def main():
    validate_mode(mode)
    ckpt_path = os.path.join(ckpt_dir, "pose_truth_autoencoder.pth")
    resume_state = None
    if mode == "resume":
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"MODE=resume requested but checkpoint not found: {ckpt_path}")
        resume_state = load_torch_checkpoint(ckpt_path, map_location=device)
        if resume_state.get("dataset") not in {None, dataset_name}:
            raise ValueError(
                f"Checkpoint dataset mismatch: ckpt={resume_state.get('dataset')} current={dataset_name}"
            )
        if int(resume_state.get("window_size", window_size)) != window_size or int(
            resume_state.get("stride", stride)
        ) != stride:
            raise ValueError(
                "Checkpoint window config mismatch: "
                f"ckpt window/stride=({resume_state.get('window_size')}, {resume_state.get('stride')}) "
                f"current=({window_size}, {stride})"
            )

    datasets = load_pose_imu_relative_datasets(
        dataset_name,
        RIDI_ROOT,
        OXIOD_ROOT,
        window_size=window_size,
        stride=stride,
        start_offset=0,
    )
    x_train = datasets["pose_train"]
    x_val = datasets["pose_val"]
    imu_train = datasets["imu_train"]
    imu_val = datasets["imu_val"]
    if x_train.shape[0] == 0 or x_val.shape[0] == 0:
        raise RuntimeError(f"No pose truth windows found for DATASET={dataset_name}")

    print(
        f"Pose truth windows: train={x_train.shape[0]} val={x_val.shape[0]} "
        f"shape={x_train.shape[1:]} imu_condition={imu_condition_mode}"
    )

    model_cfg = default_model_config() if resume_state is None else resume_state["model_config"]
    model = PoseTruthAutoEncoder(**model_cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(x_train, imu_train, shuffle=True)
    val_loader = make_loader(x_val, imu_val, shuffle=False)

    start_epoch = 0
    best_loss = float("inf")
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer_state = resume_state.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(resume_state.get("epoch", 0))
        best_loss = float(resume_state.get("best_loss", float("inf")))
        print(f"Resumed checkpoint: {ckpt_path} | start_epoch={start_epoch}")

    metrics = evaluate(model, val_loader)
    print(
        f"Initial val loss={metrics['loss']:.6f} yaw_rmse={metrics['yaw_rmse_deg']:.3f}deg "
        f"endpoint_angle={metrics['endpoint_angle_deg']:.3f}deg"
    )
    if best_loss == float("inf"):
        best_loss = metrics["loss"]
    for ep in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        total = 0.0
        count = 0
        for xb, ib in train_loader:
            xb = xb.to(device)
            ib = ib.to(device)
            recon, latent = model(xb, imu_seq=ib)
            loss_recon = quaternion_sequence_loss(recon, xb)
            loss_latent = 1e-4 * torch.mean(latent * latent)
            loss = loss_recon + loss_latent

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += float(loss_recon.item()) * xb.size(0)
            count += xb.size(0)

        train_loss = total / max(count, 1)
        metrics = evaluate(model, val_loader)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model_cfg,
                    "window_size": window_size,
                    "stride": stride,
                    "dataset": dataset_name,
                    "val_metrics": metrics,
                    "epoch": ep + 1,
                    "best_loss": best_loss,
                },
                ckpt_path,
            )

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"[PoseTruthAE Ep {ep+1}] train_loss={train_loss:.6f} val_loss={metrics['loss']:.6f} "
                f"yaw_rmse={metrics['yaw_rmse_deg']:.3f}deg "
                f"endpoint_angle={metrics['endpoint_angle_deg']:.3f}deg time={time.time()-t0:.1f}s"
            )

    best_state = load_torch_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    final_metrics = evaluate(model, val_loader)
    plot_pose_examples(model, x_val, imu_val, output_dir)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"checkpoint": ckpt_path, "metrics": final_metrics}, f, indent=2)
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
