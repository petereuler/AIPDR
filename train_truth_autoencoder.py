import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data.dense_truth import load_dense_truth_dataset
from models.truth_autoencoder import TruthAutoEncoder


WINDOW_SIZE = 64
STRIDE = 64
DATASET = "RIDI"
TRUTH_MODE = "rel_pos_body"  # "rel_pos_body" | "rel_pos_world" | "step_pos_body"

BATCH_SIZE = 512
EPOCHS = 200
LR = 2e-4
WEIGHT_DECAY = 1e-4
LATENT_DIM = 64
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RIDI_ROOT = os.path.join(PROJECT_DIR, "RIDI")
OXIOD_ROOT = os.path.join(PROJECT_DIR, "OXIOD")

dataset_name = os.getenv("DATASET", DATASET).upper()
truth_mode = os.getenv("TRUTH_MODE", TRUTH_MODE)
window_size = int(os.getenv("WINDOW_SIZE", WINDOW_SIZE))
stride = int(os.getenv("STRIDE", STRIDE))
batch_size = int(os.getenv("BATCH_SIZE", BATCH_SIZE))
epochs = int(os.getenv("EPOCHS", EPOCHS))
latent_dim = int(os.getenv("LATENT_DIM", LATENT_DIM))
mode = os.getenv("MODE", "train").lower()  # "train" | "load"

ckpt_dir = os.path.join(PROJECT_DIR, "checkpoints", dataset_name.lower())
output_dir = os.path.join(PROJECT_DIR, "output", "truth_autoencoder", dataset_name.lower(), truth_mode)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_torch_checkpoint(path, map_location):
    return torch.load(path, map_location=map_location, weights_only=False)


def model_config():
    return {
        "input_dim": 3,
        "seq_len": window_size,
        "latent_dim": latent_dim,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
    }


def normalize_truth(x, mean, std):
    return (x - mean) / std


def unnormalize_truth(x, mean, std):
    return x * std + mean


def compute_stats(x_train):
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0).astype(np.float32)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4).astype(np.float32)
    return mean.reshape(1, 1, -1), std.reshape(1, 1, -1)


def make_loader(x, shuffle):
    tensor = torch.tensor(x, dtype=torch.float32)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)


def truth_to_path(seq):
    if truth_mode == "step_pos_body":
        return np.cumsum(seq, axis=1)
    return seq


def endpoint_rmse(pred, gt):
    pred_path = truth_to_path(pred)
    gt_path = truth_to_path(gt)
    err = pred_path[:, -1, :] - gt_path[:, -1, :]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def dense_rmse(pred, gt):
    err = pred - gt
    return float(np.sqrt(np.mean(np.sum(err * err, axis=2))))


def path_length(seq):
    path = truth_to_path(seq)
    steps = np.diff(path, axis=1)
    return np.sum(np.linalg.norm(steps, axis=2), axis=1)


def evaluate(model, loader, mean_np, std_np):
    model.eval()
    losses = []
    pred_rows = []
    gt_rows = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _latent = model(xb)
            loss = F.smooth_l1_loss(recon, xb)
            losses.append(float(loss.item()) * xb.size(0))
            pred_rows.append(unnormalize_truth(recon.cpu().numpy(), mean_np, std_np))
            gt_rows.append(unnormalize_truth(xb.cpu().numpy(), mean_np, std_np))
    pred = np.concatenate(pred_rows, axis=0)
    gt = np.concatenate(gt_rows, axis=0)
    len_pred = path_length(pred)
    len_gt = path_length(gt)
    return {
        "loss": float(sum(losses) / max(len(gt), 1)),
        "dense_rmse": dense_rmse(pred, gt),
        "endpoint_rmse": endpoint_rmse(pred, gt),
        "path_len_mae": float(np.mean(np.abs(len_pred - len_gt))),
    }


def plot_recon_examples(model, x_val_norm, mean_np, std_np, out_dir, max_examples=6):
    if x_val_norm.shape[0] == 0:
        return
    model.eval()
    count = min(max_examples, x_val_norm.shape[0])
    idx = np.linspace(0, x_val_norm.shape[0] - 1, count, dtype=np.int64)
    xb = torch.tensor(x_val_norm[idx], dtype=torch.float32, device=device)
    with torch.no_grad():
        recon, _latent = model(xb)
    pred = unnormalize_truth(recon.cpu().numpy(), mean_np, std_np)
    gt = unnormalize_truth(xb.cpu().numpy(), mean_np, std_np)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150)
    axes = axes.reshape(-1)
    for ax, i in zip(axes, range(count)):
        gt_path = truth_to_path(gt[i : i + 1])[0]
        pred_path = truth_to_path(pred[i : i + 1])[0]
        ax.plot(gt_path[:, 0], gt_path[:, 1], color="black", linewidth=1.6, label="GT")
        ax.plot(pred_path[:, 0], pred_path[:, 1], color="red", linewidth=1.2, alpha=0.85, label="Recon")
        ax.scatter(gt_path[0, 0], gt_path[0, 1], color="green", s=20)
        ax.scatter(gt_path[-1, 0], gt_path[-1, 1], color="purple", s=20)
        ax.axis("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(f"Val window {int(idx[i])}")
    for ax in axes[count:]:
        ax.axis("off")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Truth reconstruction: {dataset_name} / {truth_mode}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reconstruction_examples.png"), bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(path, model, mean_np, std_np, val_metrics):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config(),
            "window_size": window_size,
            "stride": stride,
            "dataset": dataset_name,
            "truth_mode": truth_mode,
            "truth_mean": mean_np.reshape(-1).astype(np.float32),
            "truth_std": std_np.reshape(-1).astype(np.float32),
            "val_metrics": val_metrics,
        },
        path,
    )


def main():
    ckpt_path = os.path.join(ckpt_dir, f"truth_autoencoder_{truth_mode}.pth")
    x_train, x_val, _train_sequences, _val_sequences = load_dense_truth_dataset(
        dataset_name,
        RIDI_ROOT,
        OXIOD_ROOT,
        window_size=window_size,
        stride=stride,
        start_offset=0,
        truth_mode=truth_mode,
    )
    if x_train.shape[0] == 0 or x_val.shape[0] == 0:
        raise RuntimeError(f"No dense truth windows found for DATASET={dataset_name}, truth_mode={truth_mode}")

    mean_np, std_np = compute_stats(x_train)
    x_train_norm = normalize_truth(x_train, mean_np, std_np).astype(np.float32)
    x_val_norm = normalize_truth(x_val, mean_np, std_np).astype(np.float32)

    print(
        f"Dense truth windows: train={x_train.shape[0]} val={x_val.shape[0]} "
        f"shape={x_train.shape[1:]} mode={truth_mode}"
    )
    print(f"truth mean={mean_np.reshape(-1)} std={std_np.reshape(-1)}")

    model = TruthAutoEncoder(**model_config()).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(x_train_norm, shuffle=True)
    val_loader = make_loader(x_val_norm, shuffle=False)

    if mode in {"load", "resume"} and os.path.exists(ckpt_path):
        state = load_torch_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
    elif mode == "load":
        raise FileNotFoundError(f"MODE=load requested but checkpoint not found: {ckpt_path}")

    best_loss = float("inf")
    metrics = evaluate(model, val_loader, mean_np, std_np)
    print(
        f"Initial val loss={metrics['loss']:.6f} dense_rmse={metrics['dense_rmse']:.4f}m "
        f"endpoint_rmse={metrics['endpoint_rmse']:.4f}m path_len_mae={metrics['path_len_mae']:.4f}m"
    )
    if mode == "load":
        plot_recon_examples(model, x_val_norm, mean_np, std_np, output_dir)
        return

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        total = 0.0
        count = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            recon, latent = model(xb)
            loss_recon = F.smooth_l1_loss(recon, xb)
            loss_latent = 1e-4 * torch.mean(latent * latent)
            loss = loss_recon + loss_latent

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += float(loss_recon.item()) * xb.size(0)
            count += xb.size(0)

        train_loss = total / max(count, 1)
        metrics = evaluate(model, val_loader, mean_np, std_np)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(ckpt_path, model, mean_np, std_np, metrics)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"[TruthAE Ep {ep+1}] train_loss={train_loss:.6f} "
                f"val_loss={metrics['loss']:.6f} dense_rmse={metrics['dense_rmse']:.4f}m "
                f"endpoint_rmse={metrics['endpoint_rmse']:.4f}m "
                f"path_len_mae={metrics['path_len_mae']:.4f}m time={time.time()-t0:.1f}s"
            )

    best_state = load_torch_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    final_metrics = evaluate(model, val_loader, mean_np, std_np)
    plot_recon_examples(model, x_val_norm, mean_np, std_np, output_dir)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"checkpoint": ckpt_path, "metrics": final_metrics}, f, indent=2)
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
