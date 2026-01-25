import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.pose_net import PoseNetTransformer, quat_conj, quat_mul, rotate_imu, rotmat_to_quat
from models.navigator import Navigator
from utils.training_utils import load_data_2d_ridi_absheading


# ======= 参数设置 =======
window_size = 320
stride = 64
batch_size = 64
feat_dim = 64
use_joint_pose_for_nav = False
smooth_heading_labels = False
heading_sigma = 1.5

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

def train_pose_net(pose_net, train_loader):
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
            q_pred = rotmat_to_quat(R_pred)
            q_gt = yb_rel / (yb_rel.norm(dim=1, keepdim=True) + 1e-8)
            dot = torch.abs(torch.sum(q_pred * q_gt, dim=1))
            loss = torch.mean(1.0 - dot * dot)
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


def train_navigator_with_pose(pose_net, navigator, train_loader, val_loader, use_joint_pose_for_nav=False):
    params = list(navigator.parameters())
    if use_joint_pose_for_nav:
        params += list(pose_net.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    ckpt = os.path.join(ckpt_dir, "navigator.pth")
    if os.path.exists(ckpt):
        navigator.load_state_dict(torch.load(ckpt))
        print("[Navigator] 发现已有最佳模型，跳过训练")
        return

    if use_joint_pose_for_nav:
        pose_net.train()
    else:
        pose_net.eval()
    best_loss = float("inf")
    for ep in range(epochs):
        t0 = time.time()
        navigator.train()
        total = 0.0
        cnt = 0
        for batch in train_loader:
            if use_joint_pose_for_nav:
                xb, yb_dp, yb_ori, yb_rel = batch
                R_delta = pose_net(xb)
                q_anchor = quat_mul(quat_conj(yb_rel), yb_ori)
                R_anchor = quat_to_rotmat(q_anchor)
                R_abs = torch.matmul(R_anchor, R_delta)
                xb_global = rotate_imu(xb, R_abs)
                v_local = torch.matmul(R_abs.transpose(1, 2), yb_dp.unsqueeze(-1)).squeeze(-1)
                yb_dp_local = v_local[:, :2]
            else:
                xb, yb_dp, yb_ori, yb_rel = batch
                with torch.no_grad():
                    R_delta = pose_net(xb)
                    q_anchor = quat_mul(quat_conj(yb_rel), yb_ori)
                    R_anchor = quat_to_rotmat(q_anchor)
                    R_abs = torch.matmul(R_anchor, R_delta)
                    xb_global = rotate_imu(xb, R_abs)
                    v_local = torch.matmul(R_abs.transpose(1, 2), yb_dp.unsqueeze(-1)).squeeze(-1)
                    yb_dp_local = v_local[:, :2]
            pred_out = navigator(xb_global)
            gt_xy = yb_dp_local[:, :2]
            gt_len = torch.norm(gt_xy, dim=1, keepdim=True)
            gt_dir = gt_xy / (gt_len + 1e-8)
            gt_dz = v_local[:, 2:3]
            pred_dir = pred_out[:, :2]
            pred_len = pred_out[:, 2:3]
            pred_dz = pred_out[:, 3:4]
            loss_dir = F.mse_loss(pred_dir, gt_dir)
            loss_len = F.mse_loss(pred_len, gt_len)
            loss_dz = F.mse_loss(pred_dz, gt_dz)
            loss = loss_dir + loss_len + loss_dz

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
                if use_joint_pose_for_nav:
                    xb, yb_dp, yb_ori, yb_rel = batch
                    R_delta = pose_net(xb)
                    q_anchor = quat_mul(quat_conj(yb_rel), yb_ori)
                    R_anchor = quat_to_rotmat(q_anchor)
                    R_abs = torch.matmul(R_anchor, R_delta)
                    xb_global = rotate_imu(xb, R_abs)
                    v_local = torch.matmul(R_abs.transpose(1, 2), yb_dp.unsqueeze(-1)).squeeze(-1)
                    yb_dp_local = v_local[:, :2]
                else:
                    xb, yb_dp, yb_ori, yb_rel = batch
                    R_delta = pose_net(xb)
                    q_anchor = quat_mul(quat_conj(yb_rel), yb_ori)
                    R_anchor = quat_to_rotmat(q_anchor)
                    R_abs = torch.matmul(R_anchor, R_delta)
                    xb_global = rotate_imu(xb, R_abs)
                    v_local = torch.matmul(R_abs.transpose(1, 2), yb_dp.unsqueeze(-1)).squeeze(-1)
                    yb_dp_local = v_local[:, :2]
                pred_out = navigator(xb_global)
                gt_xy = yb_dp_local[:, :2]
                gt_len = torch.norm(gt_xy, dim=1, keepdim=True)
                gt_dir = gt_xy / (gt_len + 1e-8)
                gt_dz = v_local[:, 2:3]
                pred_dir = pred_out[:, :2]
                pred_len = pred_out[:, 2:3]
                pred_dz = pred_out[:, 3:4]
                loss_dir = F.mse_loss(pred_dir, gt_dir)
                loss_len = F.mse_loss(pred_len, gt_len)
                loss_dz = F.mse_loss(pred_dz, gt_dz)
                vloss = loss_dir + loss_len + loss_dz
                bs = xb.size(0)
                total_val += vloss.item() * bs
                cnt_val += bs
        val_loss = total_val / max(cnt_val, 1)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(navigator.state_dict(), ckpt)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[Navigator Ep {ep+1}] Loss: {avg_loss:.4f} | Val MSE: {val_loss:.4f} | "
                  f"Time: {time.time()-t0:.1f}s")


# ======= 主流程 =======

def main():
    print("=" * 60)
    print("PoseNet + Navigator (RIDI only)")
    print("=" * 60)

    print("\n📊 加载训练数据...")
    (x_tr, _ylen_tr, _yhead_tr, ydp_tr, yori_tr, yrel_tr, _seq_tr, _init_tr,
     x_va, _ylen_va, _yhead_va, ydp_va, yori_va, yrel_va, _seq_va, _init_va) = load_data_2d_ridi_absheading(
        ridi_root, device, window_size, stride,
        return_ori=True, return_rel_ori=True, return_seq=True, return_init=True,
        return_delta_p=True, use_abs_heading=False,
        smooth_heading=smooth_heading_labels, heading_sigma=heading_sigma
    )

    print(f"训练集: {x_tr.shape[0]} 样本")
    print(f"验证集: {x_va.shape[0]} 样本")
    if ydp_tr is not None and ydp_tr.size(1) != 3:
        raise ValueError(f"ydp_tr must be 3D, got shape {tuple(ydp_tr.shape)}")
    if ydp_va is not None and ydp_va.size(1) != 3:
        raise ValueError(f"ydp_va must be 3D, got shape {tuple(ydp_va.shape)}")

    pose_train_dataset = TensorDataset(x_tr, yrel_tr)
    if use_joint_pose_for_nav:
        nav_train_dataset = TensorDataset(x_tr, ydp_tr, yori_tr, yrel_tr)
        nav_val_dataset = TensorDataset(x_va, ydp_va, yori_va, yrel_va)
    else:
        nav_train_dataset = TensorDataset(x_tr, ydp_tr, yori_tr, yrel_tr)
        nav_val_dataset = TensorDataset(x_va, ydp_va, yori_va, yrel_va)

    pose_train_loader = DataLoader(pose_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    nav_train_loader = DataLoader(nav_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    nav_val_loader = DataLoader(nav_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    in_ch = x_tr.shape[-1]
    pose_net = PoseNetTransformer(imu_dim=6, d_model=128, nhead=4, num_layers=2, dim_feedforward=256).to(device)
    navigator = Navigator(imu_dim=in_ch, feat_dim=feat_dim).to(device)

    print("\n🎯 训练 PoseNet")
    train_pose_net(pose_net, pose_train_loader)

    print("\n🎯 训练 Navigator")
    train_navigator_with_pose(
        pose_net,
        navigator,
        nav_train_loader,
        nav_val_loader,
        use_joint_pose_for_nav=use_joint_pose_for_nav,
    )

    print("\n✅ 训练完成")
    print(f"   检查点保存在: {ckpt_dir}")


if __name__ == "__main__":
    main()
