import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from models.pose_net import PoseNet, quat_to_rotmat, rotate_imu
from models.navigator import Navigator
from utils.training_utils import load_data_2d_ridi_absheading
from utils.navigator_pipeline import accumulate_rotations, wrap_angle_torch


# ======= 参数设置 =======
window_size = 320
stride = 64
batch_size = 64
feat_dim = 64
heading_len_eps = 0.01
use_gt_pose = True

# 优化器参数
lr = 1e-4
weight_decay = 1e-4
epochs = 200
pose_epochs = 200

# 数据集
project_dir = "/home/admin407/code/zyshe/NavCorrector"
ridi_root = os.path.join(project_dir, "RIDI")
ckpt_dir = os.path.join(project_dir, "checkpoints_cls")
os.makedirs(ckpt_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ======= 训练函数 =======

def train_pose_net(pose_net, train_loader, val_loader):
    optimizer = optim.AdamW(pose_net.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt = os.path.join(ckpt_dir, "pose_net.pth")
    if os.path.exists(ckpt):
        try:
            pose_net.load_state_dict(torch.load(ckpt))
            print("[Pose] 发现已有最佳模型，跳过训练")
            return
        except Exception as e:
            print(f"[Pose] 检测到旧模型不兼容，重新训练: {e}")

    best_loss = float("inf")
    for ep in range(pose_epochs):
        t0 = time.time()
        pose_net.train()
        total = 0.0
        cnt = 0
        for xb, yb_rel in train_loader:
            R_pred = pose_net(xb)
            R_gt = quat_to_rotmat(yb_rel)
            loss = F.mse_loss(R_pred, R_gt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pose_net.parameters(), 1.0)
            optimizer.step()
            bs = xb.size(0)
            total += loss.item() * bs
            cnt += bs
        avg_loss = total / max(cnt, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(pose_net.state_dict(), ckpt)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[Pose Ep {ep+1}] Loss: {avg_loss:.4f} | Time: {time.time()-t0:.1f}s")


def train_navigator_with_pose(pose_net, navigator, train_loader, val_loader, use_gt_pose=False):
    optimizer = optim.AdamW(navigator.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt = os.path.join(ckpt_dir, "navigator.pth")
    if os.path.exists(ckpt):
        navigator.load_state_dict(torch.load(ckpt))
        print("[Navigator] 发现已有最佳模型，跳过训练")
        return

    if not use_gt_pose:
        pose_net.eval()
    best_loss = float("inf")
    for ep in range(epochs):
        t0 = time.time()
        navigator.train()
        total = 0.0
        cnt = 0
        for batch in train_loader:
            if use_gt_pose:
                xb, yb_len, yb_dp, yb_ori = batch
                with torch.no_grad():
                    R_abs = quat_to_rotmat(yb_ori)
                    xb_global = rotate_imu(xb, R_abs)
            else:
                xb, yb_len, yb_dp, seq_id, init_rot = batch
                with torch.no_grad():
                    R_delta = pose_net(xb)
                    R_abs = accumulate_rotations(R_delta, seq_id, init_rot)
                    xb_global = rotate_imu(xb, R_abs)
            pred_len, pred_h, _, pred_dp = navigator(xb_global)
            loss_dp = F.mse_loss(pred_dp, yb_dp)
            loss_len = F.mse_loss(pred_len, yb_len)
            gt_h = torch.atan2(yb_dp[:, 1], yb_dp[:, 0])
            diff_h = wrap_angle_torch(pred_h.squeeze(-1) - gt_h)
            mask = (yb_len.squeeze(-1) > heading_len_eps).float().to(device)
            loss_head = ((diff_h ** 2) * mask).sum() / (mask.sum() + 1e-8)
            w_len, w_head, w_dp = navigator.normalized_loss_weights()
            loss = w_dp * loss_dp + w_len * loss_len + w_head * loss_head

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(navigator.parameters(), 1.0)
            optimizer.step()

            bs = xb.size(0)
            total += loss.item() * bs
            cnt += bs
        avg_loss = total / max(cnt, 1)

        navigator.eval()
        total_val = 0.0
        cnt_val = 0
        with torch.no_grad():
            for batch in val_loader:
                if use_gt_pose:
                    xb, yb_len, yb_dp, yb_ori = batch
                    R_abs = quat_to_rotmat(yb_ori)
                    xb_global = rotate_imu(xb, R_abs)
                else:
                    xb, yb_len, yb_dp, seq_id, init_rot = batch
                    R_delta = pose_net(xb)
                    R_abs = accumulate_rotations(R_delta, seq_id, init_rot)
                    xb_global = rotate_imu(xb, R_abs)
                pred_len, pred_h, _, pred_dp = navigator(xb_global)
                loss_dp = F.mse_loss(pred_dp, yb_dp)
                loss_len = F.mse_loss(pred_len, yb_len)
                gt_h = torch.atan2(yb_dp[:, 1], yb_dp[:, 0])
                diff_h = wrap_angle_torch(pred_h.squeeze(-1) - gt_h)
                mask = (yb_len.squeeze(-1) > heading_len_eps).float().to(device)
                loss_head = ((diff_h ** 2) * mask).sum() / (mask.sum() + 1e-8)
                w_len, w_head, w_dp = navigator.normalized_loss_weights()
                vloss = w_dp * loss_dp + w_len * loss_len + w_head * loss_head
                bs = xb.size(0)
                total_val += vloss.item() * bs
                cnt_val += bs
        val_loss = total_val / max(cnt_val, 1)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(navigator.state_dict(), ckpt)
        if (ep + 1) % 5 == 0 or ep == 0:
            w_len, w_head, w_dp = navigator.normalized_loss_weights()
            print(f"[Navigator Ep {ep+1}] Loss: {avg_loss:.4f} | Val MSE: {val_loss:.4f} | "
                  f"W(len={w_len.item():.2f}, head={w_head.item():.2f}, dp={w_dp.item():.2f}) | "
                  f"Time: {time.time()-t0:.1f}s")


# ======= 主流程 =======

def main():
    print("=" * 60)
    print("PoseNet + Navigator (RIDI only)")
    print("=" * 60)

    print("\n📊 加载训练数据...")
    (x_tr, ylen_tr, yhead_tr, ydp_tr, yori_tr, yrel_tr, seq_tr, init_tr,
     x_va, ylen_va, yhead_va, ydp_va, yori_va, yrel_va, seq_va, init_va) = load_data_2d_ridi_absheading(
        ridi_root, device, window_size, stride,
        return_ori=True, return_rel_ori=True, return_seq=True, return_init=True,
        return_delta_p=True, use_abs_heading=False
    )

    print(f"训练集: {x_tr.shape[0]} 样本")
    print(f"验证集: {x_va.shape[0]} 样本")

    pose_train_dataset = TensorDataset(x_tr, yrel_tr)
    pose_val_dataset = TensorDataset(x_va, yrel_va)
    nav_train_dataset = TensorDataset(x_tr, ylen_tr, ydp_tr, seq_tr, init_tr)
    nav_val_dataset = TensorDataset(x_va, ylen_va, ydp_va, seq_va, init_va)

    if use_gt_pose:
        nav_train_dataset = TensorDataset(x_tr, ylen_tr, ydp_tr, yori_tr)
        nav_val_dataset = TensorDataset(x_va, ylen_va, ydp_va, yori_va)

    pose_train_loader = DataLoader(pose_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    pose_val_loader = DataLoader(pose_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    nav_train_loader = DataLoader(nav_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    nav_val_loader = DataLoader(nav_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    in_ch = x_tr.shape[-1]
    pose_net = PoseNet(imu_dim=6, hidden_dim=128).to(device)
    navigator = Navigator(imu_dim=in_ch, feat_dim=feat_dim).to(device)

    if use_gt_pose:
        print("\n🎯 使用真值姿态训练 Navigator（跳过 PoseNet）")
    else:
        print("\n🎯 训练 PoseNet")
        train_pose_net(pose_net, pose_train_loader, pose_val_loader)

    print("\n🎯 训练 Navigator")
    train_navigator_with_pose(pose_net, navigator, nav_train_loader, nav_val_loader, use_gt_pose=use_gt_pose)

    print("\n✅ 训练完成")
    print(f"   检查点保存在: {ckpt_dir}")


if __name__ == "__main__":
    main()
