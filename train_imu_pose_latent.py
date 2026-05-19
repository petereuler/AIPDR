import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data.relative_window_truth import load_pose_imu_relative_datasets
from models.imu_to_pose_latent import IMUToPoseLatent
from models.posenet import quat_to_rotmat
from models.pose_truth_autoencoder import (
    PoseTruthAutoEncoder,
    quaternion_endpoint_error_rad,
    quaternion_sequence_loss,
)


WINDOW_SIZE = 64
STRIDE = 64
DATASET = "RIDI"

BATCH_SIZE = 512
SEQUENCE_BATCH_SIZE = 128
EPOCHS = 400
LR = 2e-4
WEIGHT_DECAY = 1e-4
MODE = "retrain"  # "resume" | "retrain"
SEQUENCE_EPOCHS = 0
SEQUENCE_LEN = 8
SEQUENCE_STRIDE = 4
SEQUENCE_TRAJ_WEIGHT = 1.0
SEQUENCE_POSE_WEIGHT = 0.5
SEQUENCE_LATENT_WEIGHT = 0.25
SEQUENCE_FINAL_WEIGHT = 0.5
SEQUENCE_VAL_EVERY = 10

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RIDI_ROOT = os.path.join(PROJECT_DIR, "RIDI")
OXIOD_ROOT = os.path.join(PROJECT_DIR, "OXIOD")

dataset_name = os.getenv("DATASET", DATASET).upper()
window_size = int(os.getenv("WINDOW_SIZE", WINDOW_SIZE))
stride = int(os.getenv("STRIDE", STRIDE))
batch_size = int(os.getenv("BATCH_SIZE", BATCH_SIZE))
sequence_batch_size = int(os.getenv("SEQUENCE_BATCH_SIZE", SEQUENCE_BATCH_SIZE))
epochs = int(os.getenv("EPOCHS", EPOCHS))
mode = os.getenv("MODE", MODE).lower()
sequence_epochs = int(os.getenv("SEQUENCE_EPOCHS", SEQUENCE_EPOCHS))
sequence_len = int(os.getenv("SEQUENCE_LEN", SEQUENCE_LEN))
sequence_stride = int(os.getenv("SEQUENCE_STRIDE", SEQUENCE_STRIDE))
sequence_val_every = int(os.getenv("SEQUENCE_VAL_EVERY", SEQUENCE_VAL_EVERY))

ckpt_dir = os.path.join(PROJECT_DIR, "checkpoints", dataset_name.lower())
output_dir = os.path.join(PROJECT_DIR, "output", "imu_pose_latent", dataset_name.lower())
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def validate_mode(mode):
    if mode not in {"resume", "retrain"}:
        raise ValueError(f"MODE must be 'resume' or 'retrain', got {mode}")


def load_torch_checkpoint(path, map_location):
    return torch.load(path, map_location=map_location, weights_only=False)


def quat_to_yaw_np(q):
    q = q.detach().cpu().numpy()
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.tensor(
        __import__("numpy").arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        dtype=torch.float32,
    )


def make_loader(x_imu, y_pose, shuffle):
    ds = TensorDataset(
        torch.tensor(x_imu, dtype=torch.float32),
        torch.tensor(y_pose, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_sequence_loader(chunks, shuffle):
    imu = torch.tensor(np.stack([seq["imu"] for seq in chunks], axis=0), dtype=torch.float32)
    pose = torch.tensor(np.stack([seq["truth"] for seq in chunks], axis=0), dtype=torch.float32)
    step_disp = torch.tensor(np.stack([seq["step_disp"] for seq in chunks], axis=0), dtype=torch.float32)
    gt_traj = torch.tensor(np.stack([seq["gt_traj"] for seq in chunks], axis=0), dtype=torch.float32)
    init_rot = torch.tensor(np.stack([seq["init_rot"] for seq in chunks], axis=0), dtype=torch.float32)
    ds = TensorDataset(imu, pose, step_disp, gt_traj, init_rot)
    return DataLoader(ds, batch_size=sequence_batch_size, shuffle=shuffle)


def clone_sequence_item(seq):
    return {
        "name": seq["name"],
        "imu": seq["imu"].copy(),
        "truth": seq["truth"].copy(),
        "disp": seq["disp"].copy(),
        "step_disp": seq["step_disp"].copy(),
        "gt_traj": seq["gt_traj"].copy(),
        "init_rot": seq["init_rot"].copy(),
    }


def crop_sequence_item(seq, start_idx, seq_len):
    imu = seq["imu"]
    truth = seq["truth"]
    disp = seq["disp"]
    step_disp = seq["step_disp"]
    gt_traj = seq["gt_traj"]
    n = len(imu)
    if start_idx == 0:
        init_rot = seq["init_rot"].copy()
        gt_traj_chunk = gt_traj[: min(start_idx + seq_len, n)].copy()
    else:
        init_rot = seq["init_rot"].copy()
        gt_rel_end = quat_to_rotmat(torch.tensor(truth[:start_idx, -1, :], dtype=torch.float32))
        for idx in range(gt_rel_end.shape[0]):
            init_rot = init_rot @ gt_rel_end[idx].cpu().numpy()
        gt_traj_chunk = (gt_traj[start_idx : min(start_idx + seq_len, n)] - gt_traj[start_idx - 1]).copy()
    end_idx = min(start_idx + seq_len, n)
    return {
        "name": seq["name"],
        "imu": imu[start_idx:end_idx].copy(),
        "truth": truth[start_idx:end_idx].copy(),
        "disp": disp[start_idx:end_idx].copy(),
        "step_disp": step_disp[start_idx:end_idx].copy(),
        "gt_traj": gt_traj_chunk.astype(np.float32),
        "init_rot": init_rot.astype(np.float32),
    }


def split_sequence_item(seq, seq_len, seq_stride):
    n = len(seq["imu"])
    if n <= seq_len:
        return [clone_sequence_item(seq)]
    chunks = []
    seq_stride = int(max(1, seq_stride))
    last_start = n - seq_len
    starts = list(range(0, last_start + 1, seq_stride))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    used = set()
    for start_idx in starts:
        if start_idx in used:
            continue
        used.add(start_idx)
        chunk = crop_sequence_item(seq, start_idx, seq_len)
        if len(chunk["imu"]) == seq_len:
            chunks.append(chunk)
    return chunks


def build_predictor_config(autoencoder_state):
    pose_cfg = autoencoder_state["model_config"]
    return {
        "imu_dim": 6,
        "latent_dim": int(pose_cfg["latent_dim"]),
        "d_model": int(pose_cfg["d_model"]),
        "nhead": int(pose_cfg["nhead"]),
        "num_layers": int(pose_cfg["num_layers"]),
        "dim_feedforward": int(pose_cfg["dim_feedforward"]),
        "dropout": float(pose_cfg["dropout"]),
    }


def freeze_pose_latent_decoder_chain(autoencoder):
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False


def rollout_sequence_xy_torch(step_disp_body_seq, pose_seq, init_rot):
    if step_disp_body_seq.ndim == 3:
        step_disp_body_seq = step_disp_body_seq.unsqueeze(0)
        pose_seq = pose_seq.unsqueeze(0)
        init_rot = init_rot.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    bsz, num_windows, window_size, _ = pose_seq.shape
    rel_rot = quat_to_rotmat(pose_seq.reshape(-1, 4)).reshape(bsz, num_windows, window_size, 3, 3)
    current_r = init_rot
    pos_xy = step_disp_body_seq.new_zeros((bsz, 2))
    traj_rows = []
    for win_idx in range(num_windows):
        for step_idx in range(1, window_size):
            r_step = current_r @ rel_rot[:, win_idx, step_idx - 1]
            dp_world = (r_step @ step_disp_body_seq[:, win_idx, step_idx].unsqueeze(-1)).squeeze(-1)
            pos_xy = pos_xy + dp_world[:, :2]
        traj_rows.append(pos_xy)
        current_r = current_r @ rel_rot[:, win_idx, -1]
    traj = torch.stack(traj_rows, dim=1)
    if squeeze:
        return traj[0]
    return traj


def evaluate(predictor, autoencoder, loader):
    predictor.eval()
    autoencoder.eval()
    loss_rows = []
    yaw_rows = []
    end_rows = []
    latent_rows = []
    with torch.no_grad():
        for xb_imu, yb_pose in loader:
            xb_imu = xb_imu.to(device)
            yb_pose = yb_pose.to(device)
            z_pred = predictor(xb_imu)
            z_gt = autoencoder.encode(yb_pose)
            q_pred = autoencoder.decode(z_pred)
            loss_latent = torch.nn.functional.smooth_l1_loss(z_pred, z_gt)
            loss_pose = quaternion_sequence_loss(q_pred, yb_pose)
            loss = loss_latent + loss_pose
            loss_rows.append(float(loss.item()) * xb_imu.size(0))
            end_rows.append(float(quaternion_endpoint_error_rad(q_pred, yb_pose).item()) * xb_imu.size(0))
            yaw_pred = quat_to_yaw_np(q_pred[:, -1, :])
            yaw_gt = quat_to_yaw_np(yb_pose[:, -1, :])
            yaw_err = (yaw_pred - yaw_gt + torch.pi) % (2 * torch.pi) - torch.pi
            yaw_rows.append(float(torch.sqrt(torch.mean(yaw_err * yaw_err)).item()) * xb_imu.size(0))
            latent_rows.append(float(torch.mean(torch.norm(z_pred - z_gt, dim=1)).item()) * xb_imu.size(0))
    num = len(loader.dataset)
    return {
        "loss": float(sum(loss_rows) / max(num, 1)),
        "yaw_rmse_deg": float(sum(yaw_rows) / max(num, 1) * 180.0 / torch.pi),
        "endpoint_angle_deg": float(sum(end_rows) / max(num, 1) * 180.0 / torch.pi),
        "latent_l1": float(sum(latent_rows) / max(num, 1)),
    }


def evaluate_sequence_stage(predictor, autoencoder, loader):
    predictor.eval()
    autoencoder.eval()
    rows = []
    with torch.no_grad():
        for xb_imu, yb_pose, step_disp, gt_traj, init_rot in loader:
            xb_imu = xb_imu.to(device)
            yb_pose = yb_pose.to(device)
            step_disp = step_disp.to(device)
            gt_traj = gt_traj.to(device)
            init_rot = init_rot.to(device)
            bsz, num_seq, window_size_local, _ = xb_imu.shape
            xb_imu_flat = xb_imu.reshape(bsz * num_seq, window_size_local, -1)
            yb_pose_flat = yb_pose.reshape(bsz * num_seq, window_size_local, -1)
            z_pred = predictor(xb_imu_flat).reshape(bsz, num_seq, -1)
            z_gt = autoencoder.encode(yb_pose_flat).reshape(bsz, num_seq, -1)
            q_pred = autoencoder.decode(z_pred.reshape(bsz * num_seq, -1)).reshape(bsz, num_seq, window_size_local, -1)
            pred_traj = rollout_sequence_xy_torch(step_disp, q_pred, init_rot)

            loss_latent = F.smooth_l1_loss(z_pred, z_gt)
            loss_pose = quaternion_sequence_loss(q_pred, yb_pose)
            loss_traj = F.smooth_l1_loss(pred_traj, gt_traj)
            loss_final = F.smooth_l1_loss(pred_traj[:, -1, :], gt_traj[:, -1, :])
            yaw_pred = quat_to_yaw_np(q_pred[:, :, -1, :])
            yaw_gt = quat_to_yaw_np(yb_pose[:, :, -1, :])
            yaw_err = (yaw_pred - yaw_gt + torch.pi) % (2 * torch.pi) - torch.pi
            traj_err = pred_traj - gt_traj
            rows.append(
                {
                    "loss": float(
                        (
                            SEQUENCE_TRAJ_WEIGHT * loss_traj
                            + SEQUENCE_POSE_WEIGHT * loss_pose
                            + SEQUENCE_LATENT_WEIGHT * loss_latent
                            + SEQUENCE_FINAL_WEIGHT * loss_final
                        ).item()
                    ),
                    "traj_rmse": float(torch.sqrt(torch.mean(torch.sum(traj_err * traj_err, dim=2))).item()),
                    "yaw_rmse_deg": float(torch.sqrt(torch.mean(yaw_err * yaw_err)).item() * 180.0 / torch.pi),
                }
            )
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main():
    validate_mode(mode)
    pose_ae_ckpt = os.path.join(ckpt_dir, "pose_truth_autoencoder.pth")
    if not os.path.exists(pose_ae_ckpt):
        raise FileNotFoundError(f"Pose autoencoder checkpoint not found: {pose_ae_ckpt}")
    pose_ae_state = load_torch_checkpoint(pose_ae_ckpt, map_location=device)

    autoencoder = PoseTruthAutoEncoder(**pose_ae_state["model_config"]).to(device)
    autoencoder.load_state_dict(pose_ae_state["model_state_dict"])
    freeze_pose_latent_decoder_chain(autoencoder)

    predictor_ckpt = os.path.join(ckpt_dir, "imu_to_pose_latent.pth")
    predictor_cfg = build_predictor_config(pose_ae_state)
    resume_state = None
    if mode == "resume":
        if not os.path.exists(predictor_ckpt):
            raise FileNotFoundError(f"MODE=resume requested but checkpoint not found: {predictor_ckpt}")
        resume_state = load_torch_checkpoint(predictor_ckpt, map_location=device)

    datasets = load_pose_imu_relative_datasets(
        dataset_name,
        RIDI_ROOT,
        OXIOD_ROOT,
        window_size=window_size,
        stride=stride,
        start_offset=0,
    )
    x_train = datasets["imu_train"]
    y_train = datasets["pose_train"]
    x_val = datasets["imu_val"]
    y_val = datasets["pose_val"]
    train_sequences = datasets["train_sequences"]
    val_sequences = datasets["val_sequences"]
    if len(x_train) == 0 or len(x_val) == 0:
        raise RuntimeError(f"No aligned IMU-pose windows found for DATASET={dataset_name}")

    print(
        f"Aligned IMU->pose windows: train={x_train.shape[0]} val={x_val.shape[0]} "
        f"imu_shape={x_train.shape[1:]} pose_shape={y_train.shape[1:]}"
    )

    predictor = IMUToPoseLatent(**predictor_cfg).to(device)
    if resume_state is not None:
        predictor.load_state_dict(resume_state["model_state_dict"])
    optimizer = optim.AdamW(predictor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if resume_state is not None and "optimizer_state_dict" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])

    train_loader = make_loader(x_train, y_train, shuffle=True)
    val_loader = make_loader(x_val, y_val, shuffle=False)
    start_epoch = 0 if resume_state is None else int(resume_state.get("epoch", 0))
    best_loss = float("inf") if resume_state is None else float(resume_state.get("best_loss", float("inf")))

    metrics = evaluate(predictor, autoencoder, val_loader)
    print(
        f"Initial val loss={metrics['loss']:.6f} yaw_rmse={metrics['yaw_rmse_deg']:.3f}deg "
        f"endpoint_angle={metrics['endpoint_angle_deg']:.3f}deg latent_l1={metrics['latent_l1']:.4f}"
    )
    if best_loss == float("inf"):
        best_loss = metrics["loss"]

    for ep in range(start_epoch, epochs):
        t0 = time.time()
        predictor.train()
        total = 0.0
        count = 0
        for xb_imu, yb_pose in train_loader:
            xb_imu = xb_imu.to(device)
            yb_pose = yb_pose.to(device)
            with torch.no_grad():
                z_gt = autoencoder.encode(yb_pose)
            z_pred = predictor(xb_imu)
            q_pred = autoencoder.decode(z_pred)
            loss_latent = torch.nn.functional.smooth_l1_loss(z_pred, z_gt)
            loss_pose = quaternion_sequence_loss(q_pred, yb_pose)
            loss = loss_latent + loss_pose

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()

            total += float(loss.item()) * xb_imu.size(0)
            count += xb_imu.size(0)

        train_loss = total / max(count, 1)
        metrics = evaluate(predictor, autoencoder, val_loader)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(
                {
                    "model_state_dict": predictor.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": predictor_cfg,
                    "window_size": window_size,
                    "stride": stride,
                    "dataset": dataset_name,
                    "epoch": ep + 1,
                    "best_loss": best_loss,
                    "val_metrics": metrics,
                    "pose_autoencoder_ckpt": pose_ae_ckpt,
                },
                predictor_ckpt,
            )

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"[IMU2PoseLatent Ep {ep+1}] train_loss={train_loss:.6f} val_loss={metrics['loss']:.6f} "
                f"yaw_rmse={metrics['yaw_rmse_deg']:.3f}deg "
                f"endpoint_angle={metrics['endpoint_angle_deg']:.3f}deg latent_l1={metrics['latent_l1']:.4f} "
                f"time={time.time()-t0:.1f}s"
            )

    if SEQUENCE_EPOCHS > 0:
        print(
            f"\n[IMU2PoseLatent-Seq] sequence fine-tune start "
            f"(epochs={sequence_epochs}, seq_len={sequence_len}, seq_stride={sequence_stride})"
        )
        seq_best = float("inf")
        train_chunks = []
        val_chunks = []
        for seq in train_sequences:
            train_chunks.extend(split_sequence_item(seq, sequence_len, sequence_stride))
        for seq in val_sequences:
            val_chunks.extend(split_sequence_item(seq, sequence_len, sequence_stride))
        print(
            f"[IMU2PoseLatent-Seq] train_chunks={len(train_chunks)} val_chunks={len(val_chunks)} "
            f"sequence_batch_size={sequence_batch_size}"
        )
        val_seq_loader = make_sequence_loader(val_chunks, shuffle=False)
        seq_metrics = evaluate_sequence_stage(predictor, autoencoder, val_seq_loader)
        print(
            f"[IMU2PoseLatent-Seq Init] val_loss={seq_metrics['loss']:.6f} "
            f"traj_rmse={seq_metrics['traj_rmse']:.3f}m yaw_rmse={seq_metrics['yaw_rmse_deg']:.3f}deg"
        )
        seq_best = seq_metrics["loss"]
        for ep in range(sequence_epochs):
            predictor.train()
            train_seq_loader = make_sequence_loader(train_chunks, shuffle=True)
            train_rows = []
            t0 = time.time()
            for xb_imu, yb_pose, step_disp, gt_traj, init_rot in train_seq_loader:
                xb_imu = xb_imu.to(device)
                yb_pose = yb_pose.to(device)
                step_disp = step_disp.to(device)
                gt_traj = gt_traj.to(device)
                init_rot = init_rot.to(device)
                bsz, num_seq, window_size_local, _ = xb_imu.shape
                xb_imu_flat = xb_imu.reshape(bsz * num_seq, window_size_local, -1)
                yb_pose_flat = yb_pose.reshape(bsz * num_seq, window_size_local, -1)
                with torch.no_grad():
                    z_gt = autoencoder.encode(yb_pose_flat).reshape(bsz, num_seq, -1)
                z_pred = predictor(xb_imu_flat).reshape(bsz, num_seq, -1)
                q_pred = autoencoder.decode(z_pred.reshape(bsz * num_seq, -1)).reshape(
                    bsz, num_seq, window_size_local, -1
                )
                pred_traj = rollout_sequence_xy_torch(step_disp, q_pred, init_rot)
                loss_latent = F.smooth_l1_loss(z_pred, z_gt)
                loss_pose = quaternion_sequence_loss(q_pred, yb_pose)
                loss_traj = F.smooth_l1_loss(pred_traj, gt_traj)
                loss_final = F.smooth_l1_loss(pred_traj[:, -1, :], gt_traj[:, -1, :])
                loss = (
                    SEQUENCE_TRAJ_WEIGHT * loss_traj
                    + SEQUENCE_POSE_WEIGHT * loss_pose
                    + SEQUENCE_LATENT_WEIGHT * loss_latent
                    + SEQUENCE_FINAL_WEIGHT * loss_final
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                optimizer.step()
                train_rows.append(float(loss.item()))

            should_eval = ((ep + 1) % sequence_val_every == 0) or ep == 0 or ep == sequence_epochs - 1
            if should_eval:
                seq_metrics = evaluate_sequence_stage(predictor, autoencoder, val_seq_loader)
            else:
                seq_metrics = None
            if seq_metrics is not None and seq_metrics["loss"] < seq_best:
                seq_best = seq_metrics["loss"]
                torch.save(
                    {
                        "model_state_dict": predictor.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_config": predictor_cfg,
                        "window_size": window_size,
                        "stride": stride,
                        "dataset": dataset_name,
                        "epoch": epochs + ep + 1,
                        "best_loss": seq_best,
                        "val_metrics": seq_metrics,
                        "pose_autoencoder_ckpt": pose_ae_ckpt,
                        "sequence_finetuned": True,
                    },
                    predictor_ckpt,
                )
            if seq_metrics is not None:
                print(
                    f"[IMU2PoseLatent-Seq Ep {ep+1}] train_loss={float(np.mean(train_rows)):.6f} "
                    f"val_loss={seq_metrics['loss']:.6f} traj_rmse={seq_metrics['traj_rmse']:.3f}m "
                    f"yaw_rmse={seq_metrics['yaw_rmse_deg']:.3f}deg time={time.time()-t0:.1f}s"
                )
            elif (ep + 1) % 5 == 0 or ep == 0:
                print(
                    f"[IMU2PoseLatent-Seq Ep {ep+1}] train_loss={float(np.mean(train_rows)):.6f} "
                    f"time={time.time()-t0:.1f}s"
                )

    best_state = load_torch_checkpoint(predictor_ckpt, map_location=device)
    predictor.load_state_dict(best_state["model_state_dict"])
    final_metrics = evaluate(predictor, autoencoder, val_loader)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"checkpoint": predictor_ckpt, "metrics": final_metrics}, f, indent=2)
    print(f"Saved checkpoint to {predictor_ckpt}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
