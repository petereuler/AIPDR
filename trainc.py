import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.pose_net import PoseNetTransformer, rotmat_to_quat, quat_conj, quat_mul, quat_to_rotmat
from models.navigator import Navigator
from utils.training_utils import load_data_2d_ridi_absheading

# ======= 参数设置 =======
window_size = 64   # 约 1.6s ~ 3.2s
stride = 64
batch_size = 1024
feat_dim = 64

# 优化器参数
lr = 1e-4
weight_decay = 1e-4
epochs = 200
pose_epochs = 500
nav_epochs = 200
joint_epochs = 50
pose_output_mode = "quat"

# 路径设置
project_dir = "/home/admin407/code/zyshe/NavCorrector"
ridi_root = os.path.join(project_dir, "RIDI")
ckpt_dir = os.path.join(project_dir, "checkpoints_cls")
os.makedirs(ckpt_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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
        # 只旋转 3D 向量
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


# ======= 训练函数 =======

def train_pose_net(pose_net, train_loader):
    """Stage 1: 训练 PoseNet (保持不变)"""
    optimizer = optim.AdamW(pose_net.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt = os.path.join(ckpt_dir, "pose_net.pth")
    
    if os.path.exists(ckpt):
        print(f"[Pose] 发现检查点 {ckpt}，尝试加载...")
        try:
            pose_net.load_state_dict(torch.load(ckpt))
            print("[Pose] 加载成功")
        except:
            print("[Pose] 加载失败，重新训练")

    # 初始化 best_loss 为当前模型在训练集上的loss，确保从已训权重继续改进
    best_loss = float("inf")
    pose_net.eval()
    with torch.no_grad():
        total = 0.0
        cnt = 0
        for batch in train_loader:
            xb, yb_rel = batch[0], batch[1]
            R_pred = pose_net(xb)
            q_pred = rotmat_to_quat(R_pred)
            q_gt = yb_rel / (yb_rel.norm(dim=1, keepdim=True) + 1e-8)
            dot = torch.abs(torch.sum(q_pred * q_gt, dim=1))
            loss = torch.mean(1.0 - dot * dot)
            bs = xb.size(0)
            total += loss.item() * bs
            cnt += bs
        if cnt > 0:
            best_loss = total / cnt
    pose_net.train()
    print(f"[Pose] 开始训练... (Epochs: {pose_epochs})")
    
    for ep in range(pose_epochs):
        t0 = time.time()
        pose_net.train()
        total = 0.0
        cnt = 0
        for batch in train_loader:
            xb, yb_rel = batch[0], batch[1]
            
            # PoseNet 预测相对姿态
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
            
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[Pose Ep {ep+1}] Loss: {avg_loss:.6f} | Time: {time.time()-t0:.1f}s")


def train_navigator_corrected(
    pose_net,
    navigator,
    train_loader,
    val_loader,
    joint_finetune=False,
    pose_lr_scale=0.1,
    num_epochs=None,
    phase_name="Navigator",
):
    """
    Stage 2: 训练 Navigator (修复坐标系对齐)
    改进点：
    1. 使用 q_start (yb_init) 作为绝对锚点。
    2. R_abs = R_init @ R_rel (PoseNet预测)。
    3. 将 Linear Acc 旋转到 R_abs 定义的世界系。
    """
    if joint_finetune:
        optimizer = optim.AdamW(
            [
                {"params": navigator.parameters(), "lr": lr},
                {"params": pose_net.parameters(), "lr": lr * pose_lr_scale},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = optim.AdamW(navigator.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt = os.path.join(ckpt_dir, "navigator.pth")
    
    if joint_finetune:
        pose_net.train()
    else:
        pose_net.eval()
    
    best_loss = float("inf")
    num_epochs = epochs if num_epochs is None else num_epochs
    print(f"\n[{phase_name}] 开始训练... (Epochs: {num_epochs})")

    for ep in range(num_epochs):
        navigator.train()
        total_train_loss = 0.0
        train_cnt = 0
        
        for batch in train_loader:
            # xb: Raw IMU
            # yb_dp: GT Displacement (Global Frame)
            # yb_ori: GT Orientation at window end (For Gravity Removal / Anchor)
            # yb_rel: GT Relative Orientation (qa^-1 * qb)
            xb, yb_dp, yb_ori, yb_rel, yb_align = batch
            
            acc_raw = xb[:, :, 3:6]
            gyro_raw = xb[:, :, 0:3]
            
            # --- Step 1: 使用线性加速度（数据集提供） ---
            xb_linear = torch.cat([gyro_raw, acc_raw], dim=2)

            # --- Step 2: 绝对坐标系对齐 (Per-window anchor) ---
            if joint_finetune:
                R_rel = pose_net(xb)
            else:
                with torch.no_grad():
                    R_rel = pose_net(xb)
            # 从 qb 和 q_rel 还原 qa 作为锚点: qa = conj(q_rel) * qb
            q_anchor = quat_mul(quat_conj(yb_rel), yb_ori)
            R_anchor = quat_to_rotmat(q_anchor)
            R_abs_est = torch.matmul(R_anchor, R_rel)

            # 3. 将 Linear Acc 旋转到 "估计的世界系"
            # 这样 Navigator 看到的加速度是 Global Frame 下的 (North, East, Up)
            R_abs_aligned = torch.matmul(yb_align, R_abs_est)
            xb_aligned = rotate_imu_by_matrix(xb_linear, R_abs_aligned)
            
            # --- Step 3: Forward & Loss (world-frame dp) ---
            pred_dp_aligned = navigator(xb_aligned)
            pred_dp_world = torch.matmul(yb_align.transpose(1, 2), pred_dp_aligned.unsqueeze(-1)).squeeze(-1)
            loss = F.mse_loss(pred_dp_world, yb_dp)

            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(navigator.parameters(), 1.0)
            optimizer.step()
            
            train_cnt += xb.size(0)
            total_train_loss += loss.item() * xb.size(0)

        # --- Validation ---
        avg_train_loss = total_train_loss / max(train_cnt, 1)
        
        navigator.eval()
        total_val_loss = 0.0
        val_cnt = 0
        
        with torch.no_grad():
            for batch in val_loader:
                xb, yb_dp, yb_ori, yb_rel, yb_align = batch
                
                acc_raw = xb[:, :, 3:6]
                gyro_raw = xb[:, :, 0:3]
                xb_linear = torch.cat([gyro_raw, acc_raw], dim=2)
                
                if joint_finetune:
                    R_rel = pose_net(xb)
                else:
                    R_rel = pose_net(xb)
                q_anchor = quat_mul(quat_conj(yb_rel), yb_ori)
                R_anchor = quat_to_rotmat(q_anchor)
                R_abs_est = torch.matmul(R_anchor, R_rel)
                R_abs_aligned = torch.matmul(yb_align, R_abs_est)
                xb_aligned = rotate_imu_by_matrix(xb_linear, R_abs_aligned)
                
                pred_dp_aligned = navigator(xb_aligned)
                pred_dp_world = torch.matmul(yb_align.transpose(1, 2), pred_dp_aligned.unsqueeze(-1)).squeeze(-1)
                v_loss = F.mse_loss(pred_dp_world, yb_dp)
                         
                total_val_loss += v_loss.item() * xb.size(0)
                val_cnt += xb.size(0)
                
        avg_val_loss = total_val_loss / max(val_cnt, 1)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(navigator.state_dict(), ckpt)
            
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[{phase_name} Ep {ep+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# ======= 主流程 =======

def main():
    print("=" * 60)
    print("NavCorrector: PoseNet + Polar Navigator (Fixed Absolute Alignment)")
    print("=" * 60)

    # 1. 加载数据
    print("\n📊 加载训练数据...")
    # PoseNet: 使用带重力加速度 (acce)
    (x_tr_pose, _ylen_tr_pose, _yhead_tr_pose, yrel_tr_pose,
     x_va_pose, _ylen_va_pose, _yhead_va_pose, yrel_va_pose) = load_data_2d_ridi_absheading(
        ridi_root, device, window_size, stride,
        return_ori=False,
        return_rel_ori=True,
        return_delta_p=False,
        return_init=False,
        use_abs_heading=False,
        acc_source="acce",
        align_heading_to_init_pose=False,
    )

    # Navigator: 使用线性加速度 (linacce)
    (x_tr, _ylen_tr, _yhead_tr, ydp_tr, yori_tr, yrel_tr, _yinit_tr, yalign_tr,
     x_va, _ylen_va, _yhead_va, ydp_va, yori_va, yrel_va, _yinit_va, yalign_va) = load_data_2d_ridi_absheading(
        ridi_root, device, window_size, stride,
        return_ori=True,        # 用于去重力
        return_rel_ori=True,    # 用于 PoseNet 训练
        return_delta_p=True,    # 用于 Navigator 训练标签
        return_init=True,       # 用于 Navigator 输入对齐 (绝对姿态锚点)
        use_abs_heading=False,
        align_init_quat=True,
        align_init_quat_to_labels=False,
        return_align=True,
        acc_source="linacce",
        align_heading_to_init_pose=False,
    )
    
    # PoseNet 数据集
    pose_dataset = TensorDataset(x_tr_pose, yrel_tr_pose)
    pose_loader = DataLoader(pose_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Navigator 数据集 (新增 yinit_tr)
    nav_train_ds = TensorDataset(x_tr, ydp_tr, yori_tr, yrel_tr, yalign_tr)
    nav_val_ds = TensorDataset(x_va, ydp_va, yori_va, yrel_va, yalign_va)
    
    nav_train_loader = DataLoader(nav_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    nav_val_loader = DataLoader(nav_val_ds, batch_size=batch_size, shuffle=False)

    # 2. 初始化模型
    in_ch = x_tr.shape[-1]
    pose_net = PoseNetTransformer(imu_dim=6, d_model=128, output_mode=pose_output_mode).to(device)
    navigator = Navigator(imu_dim=in_ch, feat_dim=feat_dim).to(device)

    # 3. 训练 PoseNet
    train_pose_net(pose_net, pose_loader)

    # 4. 训练 Navigator (Stage 2: Frozen PoseNet)
    train_navigator_corrected(
        pose_net,
        navigator,
        nav_train_loader,
        nav_val_loader,
        joint_finetune=False,
        pose_lr_scale=0.1,
        num_epochs=nav_epochs,
        phase_name="Navigator-Stage2",
    )

    # 5. 训练 Navigator + PoseNet (Stage 3: Joint Finetune)
    if joint_epochs > 0:
        train_navigator_corrected(
            pose_net,
            navigator,
            nav_train_loader,
            nav_val_loader,
            joint_finetune=True,
            pose_lr_scale=0.1,
            num_epochs=joint_epochs,
            phase_name="Navigator-Stage3",
        )

    print("\n✅ 训练完成")

if __name__ == "__main__":
    main()
