import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data.dataset_OXIOD import (
    get_oxiod_predefined_split_pairs,
    load_oxiod_raw,
    window_dataset as oxiod_window,
)
from data.dataset_RIDI import load_ridi_raw, window_dataset as ridi_window
from models.posenet import PoseNetTransformer, quat_conj, quat_mul, quat_to_rotmat
from models.navigator import Navigator
from models.sequence_refiner import SequenceRefiner
from utils.training_utils import load_data_oxiod_absheading, load_data_ridi_absheading

# ======= 参数设置 =======
window_size = 64   # 约 1.6s ~ 3.2s
stride = 64
batch_size = 1024
# 可选: "RIDI" / "OXIOD"
DATASET = "RIDI"
POSENET_CONFIG = {
    "imu_dim": 6,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
}
NAVIGATOR_CONFIG = {
    "feat_dim": 64,
}

# 优化器参数
lr = 1e-4
weight_decay = 1e-4
pose_epochs = 200
nav_epochs = 100
joint_epochs = 100
stage1_mode = "resume"   # "load" | "resume" | "retrain"
stage2_mode = "load"   # "load" | "resume" | "retrain"
stage3_mode = "resume"   # "load" | "resume" | "retrain"
stage3_refiner_lr = 2e-4
stage3_refiner_wd = 1e-4
stage3_refiner_dim = 128
stage3_refiner_layers = 2
stage3_refiner_heads = 4
stage3_world_weight = 0.75
stage3_traj_weight = 0.5
stage3_final_weight = 0.1
stage3_body_weight = 0.25
stage3_pose_weight = 0.25
stage3_multi_steps = (1, 2, 4, 8, 16)
stage3_max_body_residual = 0.05
stage3_max_rot_residual_deg = 6.0

# 路径设置
project_dir = os.path.dirname(os.path.abspath(__file__))
ridi_root = os.path.join(project_dir, "RIDI")
oxiod_root = os.path.join(project_dir, "OXIOD")
dataset_name = os.getenv("DATASET", DATASET).upper()
ckpt_dir = os.path.join(project_dir, "checkpoints", dataset_name.lower())
os.makedirs(ckpt_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def build_posenet():
    return PoseNetTransformer(**POSENET_CONFIG).to(device)


def build_navigator(imu_dim):
    return Navigator(imu_dim=imu_dim, feat_dim=NAVIGATOR_CONFIG["feat_dim"]).to(device)


def checkpoint_payload(model, model_config, extra=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": dict(model_config),
        "window_size": window_size,
        "stride": stride,
        "dataset": dataset_name,
    }
    if extra:
        payload.update(extra)
    return payload


def validate_stage_mode(mode, stage_name):
    if mode not in {"load", "resume", "retrain"}:
        raise ValueError(f"{stage_name}_mode must be one of load/resume/retrain, got {mode}")


def load_ckpt_for_stage(model, ckpt_path, stage_mode, stage_name):
    validate_stage_mode(stage_mode, stage_name)
    if stage_mode in {"load", "resume"}:
        if not os.path.exists(ckpt_path):
            if stage_mode == "load":
                raise FileNotFoundError(f"[{stage_name}] requested load, but checkpoint not found: {ckpt_path}")
            print(f"[{stage_name}] checkpoint not found, will train from scratch: {ckpt_path}")
            return False
        print(f"[{stage_name}] loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        return True
    return False


def rotate_imu_by_matrix(imu_seq, R_mat):
    """
    使用旋转矩阵序列旋转 IMU 数据
    imu_seq: [B, T, 3] or [B, T, 6]
    R_mat: [B, 3, 3] (整个窗口共用一个旋转) 或 [B, T, 3, 3]
    """
    # 简单起见，假设 R_mat 是 [B, 3, 3]，即窗口的参考姿态
    # 如果 imu_seq 是 [B, T, 6] (gyro+acc)，我们通常只旋转 acc 和 gyro 向量
    # 这里我们分开处理
    batch_size, seq_len, dim = imu_seq.shape
    
    # 扩展 R_mat 到 [B, T, 3, 3]
    if R_mat.dim() == 3:
        R_seq = R_mat.unsqueeze(1).expand(batch_size, seq_len, 3, 3)
    else:
        R_seq = R_mat
        
    R_seq_flat = R_seq.reshape(-1, 3, 3)
    imu_flat = imu_seq.reshape(-1, dim)
    
    if dim == 3:
        # 只旋转三维向量
        imu_rot = torch.matmul(R_seq_flat, imu_flat.unsqueeze(-1)).squeeze(-1)
    elif dim == 6:
        # 分别旋转 Gyro (0:3) 和 Acc (3:6)
        gyro = imu_flat[:, 0:3]
        acc = imu_flat[:, 3:6]
        gyro_rot = torch.matmul(R_seq_flat, gyro.unsqueeze(-1)).squeeze(-1)
        acc_rot = torch.matmul(R_seq_flat, acc.unsqueeze(-1)).squeeze(-1)
        imu_rot = torch.cat([gyro_rot, acc_rot], dim=1)
    else:
        raise ValueError("Unsupported IMU dim")
        
    return imu_rot.view(batch_size, seq_len, dim)


def compute_init_rot(ori, pos_xyz, window_size, stride):
    max_start = pos_xyz.shape[0] - window_size - 1
    if max_start <= 0:
        return np.zeros((0, 3, 3), dtype=np.float32)
    a_indices = []
    for idx in range(0, max_start, stride):
        a = idx + window_size // 2 - stride // 2
        a = max(0, min(a, len(ori) - 1))
        a_indices.append(a)
    if not a_indices:
        return np.zeros((0, 3, 3), dtype=np.float32)
    q0 = ori[a_indices[0]].astype(np.float32)
    w, x, y, z = q0
    init_rot = np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ],
        dtype=np.float32,
    )
    return np.repeat(init_rot[None, :, :], len(a_indices), axis=0)


def _read_name_list(path):
    with open(path, "r") as f:
        return [line.strip().split(",")[0] for line in f if line.strip()]


def build_ridi_sequence_items(root, names, device, window_size, stride):
    data_root = os.path.join(root, "data")
    items = []
    for name in names:
        seq_dir = os.path.join(data_root, name)
        if not os.path.isdir(seq_dir):
            continue
        gyro_pose, acc_pose, pos_xyz, ori = load_ridi_raw(seq_dir, acc_source="acce")
        gyro_nav, acc_nav, _pos_nav, _ori_nav = load_ridi_raw(seq_dir, acc_source="linacce")

        [gx_nav, ax_nav], [dl, yori, yrel, ydp, ydp_world], _init_pos, _init_head = ridi_window(
            gyro_nav,
            acc_nav,
            pos_xyz,
            ori,
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
            gyro_pose,
            acc_pose,
            pos_xyz,
            ori,
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
        init_rot_np = compute_init_rot(ori, pos_xyz, window_size, stride)
        if init_rot_np.shape[0] == 0:
            continue

        m = min(gx_pose.shape[0], gx_nav.shape[0], len(ydp), len(ydp_world), len(yrel))
        items.append(
            {
                "name": name,
                "gx_pose": gx_pose[:m],
                "ax_pose": ax_pose[:m],
                "gx_nav": gx_nav[:m],
                "ax_nav": ax_nav[:m],
                "x_pose": torch.tensor(
                    np.concatenate([gx_pose[:m], ax_pose[:m]], axis=-1),
                    dtype=torch.float32,
                    device=device,
                ),
                "x_nav": torch.tensor(
                    np.concatenate([gx_nav[:m], ax_nav[:m]], axis=-1),
                    dtype=torch.float32,
                    device=device,
                ),
                "y_dp_body": torch.tensor(ydp[:m], dtype=torch.float32, device=device),
                "y_dp_world": torch.tensor(ydp_world[:m], dtype=torch.float32, device=device),
                "y_rel": torch.tensor(yrel[:m], dtype=torch.float32, device=device),
                "init_rot": torch.tensor(init_rot_np[0], dtype=torch.float32, device=device),
            }
        )
    return items


def build_oxiod_sequence_items(root, pairs, device, window_size, stride):
    items = []
    for name, imu_file, gt_file in pairs:
        gyro, acc, pos_xyz, ori = load_oxiod_raw(imu_file, gt_file)
        [gx_nav, ax_nav], [dl, yori, yrel, ydp, ydp_world], _init_pos, _init_head = oxiod_window(
            gyro,
            acc,
            pos_xyz,
            ori,
            window_size=window_size,
            stride=stride,
            filter_window=0,
            smooth_length=False,
            return_ori=True,
            return_rel_ori=True,
            return_delta_p=True,
            return_delta_p_world=True,
        )
        [gx_pose, ax_pose], _labels_pose, _init_pos_pose, _init_head_pose = oxiod_window(
            gyro,
            acc,
            pos_xyz,
            ori,
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
        init_rot_np = compute_init_rot(ori, pos_xyz, window_size, stride)
        if init_rot_np.shape[0] == 0:
            continue

        m = min(gx_pose.shape[0], gx_nav.shape[0], len(ydp), len(ydp_world), len(yrel))
        items.append(
            {
                "name": name,
                "gx_pose": gx_pose[:m],
                "ax_pose": ax_pose[:m],
                "gx_nav": gx_nav[:m],
                "ax_nav": ax_nav[:m],
                "x_pose": torch.tensor(
                    np.concatenate([gx_pose[:m], ax_pose[:m]], axis=-1),
                    dtype=torch.float32,
                    device=device,
                ),
                "x_nav": torch.tensor(
                    np.concatenate([gx_nav[:m], ax_nav[:m]], axis=-1),
                    dtype=torch.float32,
                    device=device,
                ),
                "y_dp_body": torch.tensor(ydp[:m], dtype=torch.float32, device=device),
                "y_dp_world": torch.tensor(ydp_world[:m], dtype=torch.float32, device=device),
                "y_rel": torch.tensor(yrel[:m], dtype=torch.float32, device=device),
                "init_rot": torch.tensor(init_rot_np[0], dtype=torch.float32, device=device),
            }
        )
    return items


def load_sequence_items(dataset_name, device, window_size, stride):
    if dataset_name == "RIDI":
        train_list = os.path.join(ridi_root, "data", "list_train_publish_v2.txt")
        val_list = os.path.join(ridi_root, "data", "list_test_publish_v2.txt")
        train_names = _read_name_list(train_list)
        val_names = _read_name_list(val_list)
        return (
            build_ridi_sequence_items(ridi_root, train_names, device, window_size, stride),
            build_ridi_sequence_items(ridi_root, val_names, device, window_size, stride),
        )
    if dataset_name == "OXIOD":
        train_pairs = get_oxiod_predefined_split_pairs(oxiod_root, split="train", sensor="syn")
        val_pairs = get_oxiod_predefined_split_pairs(oxiod_root, split="test", sensor="syn")
        return (
            build_oxiod_sequence_items(oxiod_root, train_pairs, device, window_size, stride),
            build_oxiod_sequence_items(oxiod_root, val_pairs, device, window_size, stride),
        )
    raise ValueError(f"Unsupported DATASET={dataset_name}, expected RIDI or OXIOD")


def quaternion_alignment_loss(q_pred, q_gt):
    q_gt = q_gt / (q_gt.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=1))
    return torch.mean(1.0 - dot)


def so3_exp_torch(rotvec):
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = rotvec / theta
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    zero = torch.zeros_like(x)
    K = torch.cat(
        [
            torch.cat([zero, -z, y], dim=-1)[..., None, :],
            torch.cat([z, zero, -x], dim=-1)[..., None, :],
            torch.cat([-y, x, zero], dim=-1)[..., None, :],
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(rotvec.shape[:-1] + (3, 3))
    sin_term = torch.sin(theta)[..., None]
    cos_term = (1.0 - torch.cos(theta))[..., None]
    return eye + sin_term * K + cos_term * (K @ K)


def build_sequence_tokens(posenet, navigator, seq_item):
    pose_feat = posenet.encode_features(seq_item["x_pose"])
    nav_feat = navigator.encode_features(seq_item["x_nav"])
    q_rel_pred = posenet(seq_item["x_pose"])
    dp_body_pred = navigator(seq_item["x_nav"])
    q_gt = seq_item["y_rel"] / (seq_item["y_rel"].norm(dim=1, keepdim=True) + 1e-8)
    q_pred = q_rel_pred / (q_rel_pred.norm(dim=1, keepdim=True) + 1e-8)
    quat_dot = torch.sum(q_pred * q_gt, dim=1, keepdim=True)
    quat_dot = torch.where(quat_dot < 0, -quat_dot, quat_dot)
    token = torch.cat(
        [
            pose_feat,
            nav_feat,
            dp_body_pred,
            q_rel_pred,
            torch.linalg.norm(dp_body_pred[:, :2], dim=1, keepdim=True),
            quat_dot,
        ],
        dim=1,
    )
    return token, q_rel_pred, dp_body_pred


def rollout_with_refiner(posenet, navigator, refiner, seq_item):
    with torch.no_grad():
        tokens, q_rel_base, dp_body_base = build_sequence_tokens(posenet, navigator, seq_item)
    body_residual, rot_residual = refiner(tokens)
    dp_body_pred = dp_body_base + body_residual
    r_base = quat_to_rotmat(q_rel_base)
    r_corr = so3_exp_torch(rot_residual)
    r_delta = torch.matmul(r_base, r_corr)
    current_r = seq_item["init_rot"]
    pred_world = []
    pred_traj = []
    pos_xy = seq_item["x_nav"].new_zeros(2)
    for idx in range(dp_body_pred.size(0)):
        dp_world = torch.matmul(current_r, dp_body_pred[idx].unsqueeze(-1)).squeeze(-1)
        pred_world.append(dp_world)
        pos_xy = pos_xy + dp_world[:2]
        pred_traj.append(pos_xy)
        current_r = torch.matmul(current_r, r_delta[idx])
    return {
        "q_rel_base": q_rel_base,
        "dp_body_base": dp_body_base,
        "dp_body_pred": dp_body_pred,
        "pred_world": torch.stack(pred_world, dim=0),
        "pred_traj": torch.stack(pred_traj, dim=0),
        "rot_residual": rot_residual,
        "body_residual": body_residual,
    }


def sequence_rollout_loss(posenet, navigator, refiner, seq_item):
    out = rollout_with_refiner(posenet, navigator, refiner, seq_item)
    gt_traj = torch.cumsum(seq_item["y_dp_world"][:, :2], dim=0)
    loss_pose = quaternion_alignment_loss(out["q_rel_base"], seq_item["y_rel"])
    loss_body = F.smooth_l1_loss(out["dp_body_pred"], seq_item["y_dp_body"])
    loss_world = F.smooth_l1_loss(out["pred_world"], seq_item["y_dp_world"])
    traj_terms = [F.smooth_l1_loss(out["pred_traj"], gt_traj)]
    for step in stage3_multi_steps:
        if out["pred_traj"].size(0) > step:
            traj_terms.append(
                F.smooth_l1_loss(
                    out["pred_traj"][step:] - out["pred_traj"][:-step],
                    gt_traj[step:] - gt_traj[:-step],
                )
            )
    loss_traj = sum(traj_terms) / len(traj_terms)
    loss_final = F.smooth_l1_loss(out["pred_traj"][-1], gt_traj[-1])
    loss = (
        stage3_world_weight * loss_world
        + stage3_traj_weight * loss_traj
        + stage3_final_weight * loss_final
        + stage3_body_weight * loss_body
        + stage3_pose_weight * loss_pose
    )
    with torch.no_grad():
        ate = torch.sqrt(torch.mean(torch.sum((out["pred_traj"] - gt_traj) ** 2, dim=1)))
    return loss, {
        "loss": float(loss.item()),
        "world": float(loss_world.item()),
        "traj": float(loss_traj.item()),
        "final": float(loss_final.item()),
        "body": float(loss_body.item()),
        "pose": float(loss_pose.item()),
        "ate": float(ate.item()),
    }


def eval_sequence_stage3(posenet, navigator, refiner, seq_items):
    posenet.eval()
    navigator.eval()
    refiner.eval()
    stats = []
    with torch.no_grad():
        for item in seq_items:
            _loss, info = sequence_rollout_loss(posenet, navigator, refiner, item)
            stats.append(info)
    return {key: float(np.mean([row[key] for row in stats])) for key in stats[0]}


def train_sequence_stage3(posenet, navigator, train_seq_items, val_seq_items, num_epochs):
    posenet.eval()
    navigator.eval()
    for param in posenet.parameters():
        param.requires_grad = False
    for param in navigator.parameters():
        param.requires_grad = False

    sample_item = train_seq_items[0]
    token_dim = (
        posenet.encode_features(sample_item["x_pose"][:1]).shape[1]
        + navigator.encode_features(sample_item["x_nav"][:1]).shape[1]
        + 3
        + 4
        + 1
        + 1
    )
    refiner = SequenceRefiner(
        input_dim=token_dim,
        d_model=stage3_refiner_dim,
        nhead=stage3_refiner_heads,
        num_layers=stage3_refiner_layers,
        dim_feedforward=stage3_refiner_dim * 4,
        max_body_residual=stage3_max_body_residual,
        max_rot_residual_deg=stage3_max_rot_residual_deg,
    ).to(device)
    optimizer = optim.AdamW(refiner.parameters(), lr=stage3_refiner_lr, weight_decay=stage3_refiner_wd)

    pose_ckpt = os.path.join(ckpt_dir, "posenet.pth")
    nav_ckpt = os.path.join(ckpt_dir, "navigator.pth")
    refiner_ckpt = os.path.join(ckpt_dir, "sequence_refiner.pth")
    if stage3_mode in {"load", "resume"} and os.path.exists(refiner_ckpt):
        print(f"[Stage3-Refiner] loading checkpoint: {refiner_ckpt}")
        refiner_data = torch.load(refiner_ckpt, map_location=device)
        refiner.load_state_dict(refiner_data["model_state_dict"])
    elif stage3_mode == "load":
        raise FileNotFoundError(f"[Stage3-Refiner] requested load, but checkpoint not found: {refiner_ckpt}")

    best_stats = eval_sequence_stage3(posenet, navigator, refiner, val_seq_items)
    best_loss = best_stats["loss"]
    if stage3_mode == "load":
        print("[Navigator-Stage3-Sequence] stage3_mode=load, skip training.")
        return
    print(
        f"\n[Navigator-Stage3-Sequence] 开始训练... (Epochs: {num_epochs}) "
        f"| init val_loss={best_loss:.4f} val_ate={best_stats['ate']:.3f}"
    )

    for ep in range(num_epochs):
        refiner.train()
        order = np.random.permutation(len(train_seq_items))
        train_stats = []
        for idx in order:
            loss, info = sequence_rollout_loss(posenet, navigator, refiner, train_seq_items[int(idx)])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            optimizer.step()
            train_stats.append(info)

        train_mean = {key: float(np.mean([row[key] for row in train_stats])) for key in train_stats[0]}
        val_stats = eval_sequence_stage3(posenet, navigator, refiner, val_seq_items)
        if val_stats["loss"] < best_loss:
            best_loss = val_stats["loss"]
            torch.save(
                {
                    "input_dim": token_dim,
                    "model_state_dict": refiner.state_dict(),
                    "window_size": window_size,
                    "stride": stride,
                    "dataset": dataset_name,
                    "config": {
                        "d_model": stage3_refiner_dim,
                        "nhead": stage3_refiner_heads,
                        "num_layers": stage3_refiner_layers,
                        "dim_feedforward": stage3_refiner_dim * 4,
                        "max_body_residual": stage3_max_body_residual,
                        "max_rot_residual_deg": stage3_max_rot_residual_deg,
                    },
                },
                refiner_ckpt,
            )
            torch.save(checkpoint_payload(posenet, POSENET_CONFIG), pose_ckpt)
            torch.save(checkpoint_payload(navigator, {"imu_dim": 6, **NAVIGATOR_CONFIG}), nav_ckpt)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"[Navigator-Stage3-Sequence Ep {ep+1}] "
                f"lr_refiner={optimizer.param_groups[0]['lr']:.1e} "
                f"Train loss={train_mean['loss']:.4f} ate={train_mean['ate']:.3f} | "
                f"Val loss={val_stats['loss']:.4f} ate={val_stats['ate']:.3f}"
            )


# ======= 训练函数 =======

def train_posenet(posenet, train_loader):
    """Stage 1: 训练 posenet。"""
    optimizer = optim.AdamW(posenet.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt = os.path.join(ckpt_dir, "posenet.pth")
    loaded = load_ckpt_for_stage(posenet, ckpt, stage1_mode, "Stage1-Pose")
    if stage1_mode == "load":
        print("[Pose] stage1_mode=load, skip training.")
        return

    # 先用当前权重在训练集上估计基线 loss，便于从已有 checkpoint 继续训练。
    best_loss = float("inf")
    if loaded:
        posenet.eval()
        with torch.no_grad():
            total = 0.0
            cnt = 0
            for batch in train_loader:
                xb, yb_rel = batch[0], batch[1]
                q_fused = posenet(xb)
                q_gt = yb_rel / (yb_rel.norm(dim=1, keepdim=True) + 1e-8)
                dot = torch.abs(torch.sum(q_fused * q_gt, dim=1))
                loss = torch.mean(1.0 - dot)
                bs = xb.size(0)
                total += loss.item() * bs
                cnt += bs
            if cnt > 0:
                best_loss = total / cnt
    posenet.train()
    print(f"[Pose] 开始训练... (Epochs: {pose_epochs})")
    
    for ep in range(pose_epochs):
        t0 = time.time()
        posenet.train()
        total = 0.0
        cnt = 0
        for batch in train_loader:
            xb, yb_rel = batch[0], batch[1]
            q_fused = posenet(xb)
            q_gt = yb_rel / (yb_rel.norm(dim=1, keepdim=True) + 1e-8)
            dot = torch.abs(torch.sum(q_fused * q_gt, dim=1))
            loss = torch.mean(1.0 - dot)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(posenet.parameters(), 1.0)
            optimizer.step()
            
            bs = xb.size(0)
            total += loss.item() * bs
            cnt += bs
            
        avg_loss = total / max(cnt, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint_payload(posenet, POSENET_CONFIG), ckpt)
            
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[Pose Ep {ep+1}] Loss: {avg_loss:.6f} | Time: {time.time()-t0:.1f}s")


def train_navigator(
    posenet,
    navigator,
    train_loader,
    val_loader,
    joint_finetune=False,
    pose_lr_scale=0.1,
    num_epochs=None,
    phase_name="Navigator",
):
    """训练 navigator，可选联合微调 posenet。"""
    if joint_finetune:
        raise ValueError("Stage3 joint finetune is handled by train_sequence_stage3(...)")
    else:
        optimizer = optim.AdamW(navigator.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt = os.path.join(ckpt_dir, "navigator.pth")
    loaded = load_ckpt_for_stage(navigator, ckpt, stage2_mode, "Stage2-Nav")
    if stage2_mode == "load":
        print(f"[{phase_name}] stage2_mode=load, skip training.")
        return
    
    posenet.eval()
    
    best_loss = float("inf")
    if loaded:
        navigator.eval()
        total_val_loss = 0.0
        val_cnt = 0
        with torch.no_grad():
            for batch in val_loader:
                xb, yb_dp, _yb_dpw, _yb_ori, _yb_rel, _yb_align = batch
                acc_raw = xb[:, :, 3:6]
                gyro_raw = xb[:, :, 0:3]
                xb_linear = torch.cat([gyro_raw, acc_raw], dim=2)
                pred_dp_body = navigator(xb_linear)
                v_loss_body = F.mse_loss(pred_dp_body, yb_dp)
                total_val_loss += v_loss_body.item() * xb.size(0)
                val_cnt += xb.size(0)
        if val_cnt > 0:
            best_loss = total_val_loss / val_cnt
    num_epochs = nav_epochs if num_epochs is None else num_epochs
    print(f"\n[{phase_name}] 开始训练... (Epochs: {num_epochs})")

    for ep in range(num_epochs):
        navigator.train()
        total_train_loss = 0.0
        train_cnt = 0
        
        for batch in train_loader:
            # xb: IMU 窗口
            # yb_dp: 手机坐标系位移标签
            # yb_dpw: 世界坐标系位移标签
            # yb_ori / yb_rel: 用于联合训练时的姿态监督
            xb, yb_dp, yb_dpw, yb_ori, yb_rel, yb_align = batch
            
            acc_raw = xb[:, :, 3:6]
            gyro_raw = xb[:, :, 0:3]
            
            # Step 1: 使用数据集提供的线性加速度，输入保持在手机坐标系。
            xb_linear = torch.cat([gyro_raw, acc_raw], dim=2)

            # Step 2: navigator 预测手机坐标系位移。
            pred_dp_body = navigator(xb_linear)

            # Step 3: 联合训练时再通过姿态把 body 位移转回世界系做监督。
            dp_body_gt = yb_dp
            loss_body = F.mse_loss(pred_dp_body, dp_body_gt)
            loss = loss_body

            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(navigator.parameters(), 1.0)
            optimizer.step()
            
            train_cnt += xb.size(0)
            total_train_loss += loss.item() * xb.size(0)

        # Validation
        avg_train_loss = total_train_loss / max(train_cnt, 1)
        
        navigator.eval()
        total_val_loss = 0.0
        val_cnt = 0
        
        with torch.no_grad():
            for batch in val_loader:
                xb, yb_dp, yb_dpw, yb_ori, yb_rel, yb_align = batch
                
                acc_raw = xb[:, :, 3:6]
                gyro_raw = xb[:, :, 0:3]
                xb_linear = torch.cat([gyro_raw, acc_raw], dim=2)
                
                R_rel = quat_to_rotmat(posenet(xb))
                q_anchor = quat_mul(quat_conj(yb_rel), yb_ori)
                R_anchor = quat_to_rotmat(q_anchor)
                R_abs_est = torch.matmul(R_anchor, R_rel)
                R_abs_aligned = torch.matmul(yb_align, R_abs_est)
                xb_aligned = rotate_imu_by_matrix(xb_linear, R_abs_aligned)
                
                pred_dp_body = navigator(xb_linear)
                dp_body_gt = yb_dp
                v_loss_body = F.mse_loss(pred_dp_body, dp_body_gt)
                v_loss = v_loss_body
                         
                total_val_loss += v_loss.item() * xb.size(0)
                val_cnt += xb.size(0)
                
        avg_val_loss = total_val_loss / max(val_cnt, 1)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(checkpoint_payload(navigator, {"imu_dim": 6, **NAVIGATOR_CONFIG}), ckpt)
            
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[{phase_name} Ep {ep+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# ======= 主流程 =======

def main():
    print("=" * 60)
    print(f"NavCorrector: PoseNet + Polar Navigator (Dataset={dataset_name})")
    print("=" * 60)

    # 1. 加载数据
    print("\n📊 加载训练数据...")
    if dataset_name == "RIDI":
        # posenet 使用带重力加速度 (acce)
        (x_tr_pose, _ylen_tr_pose, yrel_tr_pose,
         x_va_pose, _ylen_va_pose, yrel_va_pose) = load_data_ridi_absheading(
            ridi_root, device, window_size, stride,
            return_ori=False,
            return_rel_ori=True,
            return_delta_p=False,
            return_init=False,
            acc_source="acce",
        )

        # navigator 使用线性加速度 (linacce)
        (x_tr, _ylen_tr, ydp_tr, ydpw_tr, yori_tr, yrel_tr, _yinit_tr, yalign_tr,
         x_va, _ylen_va, ydp_va, ydpw_va, yori_va, yrel_va, _yinit_va, yalign_va) = load_data_ridi_absheading(
            ridi_root, device, window_size, stride,
            return_ori=True,
            return_rel_ori=True,
            return_delta_p=True,
            return_delta_p_world=True,
            return_init=True,
            align_init_quat=True,
            align_init_quat_to_labels=False,
            return_align=True,
            acc_source="linacce",
        )
    elif dataset_name == "OXIOD":
        # OXIOD 里 posenet 和 navigator 使用同一输入源。
        (x_tr_pose, _ylen_tr_pose, yrel_tr_pose,
         x_va_pose, _ylen_va_pose, yrel_va_pose) = load_data_oxiod_absheading(
            oxiod_root, device, window_size, stride,
            return_ori=False,
            return_rel_ori=True,
            return_delta_p=False,
            return_init=False,
        )
        (x_tr, _ylen_tr, ydp_tr, ydpw_tr, yori_tr, yrel_tr, _yinit_tr, yalign_tr,
         x_va, _ylen_va, ydp_va, ydpw_va, yori_va, yrel_va, _yinit_va, yalign_va) = load_data_oxiod_absheading(
            oxiod_root, device, window_size, stride,
            return_ori=True,
            return_rel_ori=True,
            return_delta_p=True,
            return_delta_p_world=True,
            return_init=True,
            align_init_quat=True,
            align_init_quat_to_labels=False,
            return_align=True,
        )
    else:
        raise ValueError(f"Unsupported DATASET={dataset_name}, expected RIDI or OXIOD")
    
    # posenet 数据集
    pose_dataset = TensorDataset(x_tr_pose, yrel_tr_pose)
    pose_loader = DataLoader(pose_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # navigator 数据集
    nav_train_ds = TensorDataset(x_tr, ydp_tr, ydpw_tr, yori_tr, yrel_tr, yalign_tr)
    nav_val_ds = TensorDataset(x_va, ydp_va, ydpw_va, yori_va, yrel_va, yalign_va)
    
    nav_train_loader = DataLoader(nav_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    nav_val_loader = DataLoader(nav_val_ds, batch_size=batch_size, shuffle=False)

    # 2. 初始化模型
    in_ch = x_tr.shape[-1]
    posenet = build_posenet()
    navigator = build_navigator(in_ch)

    # 3. 训练 posenet
    train_posenet(posenet, pose_loader)

    # 4. 训练 navigator（冻结 posenet）
    train_navigator(
        posenet,
        navigator,
        nav_train_loader,
        nav_val_loader,
        joint_finetune=False,
        pose_lr_scale=0.1,
        num_epochs=nav_epochs,
        phase_name="Navigator-Stage2",
    )

    # 5. 联合微调 navigator + posenet
    if joint_epochs > 0:
        print("\n📚 加载 Stage3 序列数据...")
        seq_train_items, seq_val_items = load_sequence_items(dataset_name, device, window_size, stride)
        print(f"[Stage3] Train seq={len(seq_train_items)} | Val seq={len(seq_val_items)}")
        train_sequence_stage3(
            posenet,
            navigator,
            seq_train_items,
            seq_val_items,
            num_epochs=joint_epochs,
        )

    print("\n✅ 训练完成")

if __name__ == "__main__":
    main()
