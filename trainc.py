import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.pose_net import PoseNetTransformer, quat_conj, quat_mul, rotate_imu, rotmat_to_quat, quat_to_rotmat
from models.navigator import Navigator
from utils.training_utils import load_data_2d_ridi_absheading

# ======= 参数设置 =======
window_size = 320   # 约 1.6s ~ 3.2s，取决于采样率，保证覆盖一个步态
stride = 64
batch_size = 64
feat_dim = 64

# 优化器参数
lr = 1e-4
weight_decay = 1e-4
epochs = 200
pose_epochs = 200

# 物理常量
GRAVITY = 9.81  # 重力加速度 m/s^2

# 路径设置 (请根据你的实际路径修改)
project_dir = "/home/admin407/code/zyshe/NavCorrector"
ridi_root = os.path.join(project_dir, "RIDI")
ckpt_dir = os.path.join(project_dir, "checkpoints_cls")
os.makedirs(ckpt_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ======= 工具函数：去重力 =======

def remove_gravity(acc_raw, q_gt):
    """
    从原始加速度中扣除重力，得到线性加速度。
    acc_raw: [B, T, 3] (Body Frame)
    q_gt: [B, T, 4] (Body -> World Orientation, scalar last/first depending on format)
          这里假设 q_gt 是 RIDI 格式，我们在 dataset 里处理过可能是 wxyz
    Returns:
        acc_linear: [B, T, 3]
    """
    # 1. 构造世界系重力向量 [0, 0, 9.81]
    batch_size, seq_len, _ = acc_raw.shape
    g_world = torch.tensor([0.0, 0.0, GRAVITY], device=acc_raw.device).view(1, 1, 3)
    g_world = g_world.expand(batch_size, seq_len, 3)
    
    # 2. 将重力转到 Body Frame: g_body = R_world2body @ g_world
    # R_world2body 就是 q_gt 的共轭 (如果 q_gt 是 Body->World)
    # 假设 models.pose_net.quat_to_rotmat 接受的是 [w, x, y, z]
    # 需要先将 q_gt reshape 成 [B*T, 4] 处理再变回来
    q_flat = q_gt.reshape(-1, 4)
    R_b2w = quat_to_rotmat(q_flat) # [B*T, 3, 3]
    R_w2b = R_b2w.transpose(1, 2)
    
    g_world_flat = g_world.reshape(-1, 3).unsqueeze(-1) # [B*T, 3, 1]
    g_body_flat = torch.matmul(R_w2b, g_world_flat).squeeze(-1) # [B*T, 3]
    g_body = g_body_flat.view(batch_size, seq_len, 3)
    
    # 3. 扣除重力
    acc_linear = acc_raw - g_body
    return acc_linear


# ======= 训练函数 =======

def train_pose_net(pose_net, train_loader):
    """Stage 1: 训练 PoseNet (保持不变)"""
    optimizer = optim.AdamW(pose_net.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt = os.path.join(ckpt_dir, "pose_net.pth")
    
    # 如果有预训练模型可以选择加载，这里为了演示保留训练逻辑
    if os.path.exists(ckpt):
        print(f"[Pose] 发现检查点 {ckpt}，尝试加载...")
        try:
            pose_net.load_state_dict(torch.load(ckpt))
            print("[Pose] 加载成功")
        except:
            print("[Pose] 加载失败，重新训练")

    best_loss = float("inf")
    print(f"[Pose] 开始训练... (Epochs: {pose_epochs})")
    
    for ep in range(pose_epochs):
        t0 = time.time()
        pose_net.train()
        total = 0.0
        cnt = 0
        # PoseNet 只需要 xb (raw imu) 和 yb_rel (相对旋转真值)
        for batch in train_loader:
            xb, yb_rel = batch[0], batch[1]
            
            R_pred = pose_net(xb)
            q_pred = rotmat_to_quat(R_pred)
            
            # 简单的余弦距离 Loss
            q_gt = yb_rel / (yb_rel.norm(dim=1, keepdim=True) + 1e-8)
            dot = torch.abs(torch.sum(q_pred * q_gt, dim=1))
            loss = torch.mean(1.0 - dot * dot) # 1 - cos^2
            
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


def train_navigator_corrected(pose_net, navigator, train_loader, val_loader):
    """
    Stage 2: 训练 Navigator (Core Modification)
    策略：
    1. 去重力 (Using GT Orientation for clean input)
    2. 坐标系对齐 (Using PoseNet Prediction)
    3. 残差学习 (Target is inverse-rotated displacement)
    """
    optimizer = optim.AdamW(navigator.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt = os.path.join(ckpt_dir, "navigator.pth")
    
    pose_net.eval() # 固定 PoseNet
    
    best_loss = float("inf")
    print(f"\n[Navigator] 开始训练... (Epochs: {epochs})")

    for ep in range(epochs):
        t0 = time.time()
        navigator.train()
        total_train_loss = 0.0
        train_cnt = 0
        
        for batch in train_loader:
            # Unpack data
            # xb: Raw IMU [B, T, 6] (Gyro, Acc)
            # yb_dp: GT Displacement World [B, 3] (dx, dy, dz)
            # yb_ori: GT Orientation Sequence [B, T, 4] (用于去重力)
            xb, yb_dp, yb_ori = batch
            
            acc_raw = xb[:, :, 3:6]
            gyro_raw = xb[:, :, 0:3]
            
            # --- Step 1: 物理去重力 (Physics-based Preprocessing) ---
            # 使用真值姿态去重力，保证输入给 Navigator 的是纯净的运动加速度
            # (在 Inference 时，这一步可以使用 PoseNet 的预测姿态，或者互补滤波结果)
            with torch.no_grad():
                acc_linear = remove_gravity(acc_raw, yb_ori)
            
            # 构造去重力后的 IMU 序列 (Gyro 不变, Acc 变了)
            xb_linear = torch.cat([gyro_raw, acc_linear], dim=2)

            # --- Step 2: 获取 PoseNet 预测的姿态基准 ---
            with torch.no_grad():
                # R_pred: [B, 3, 3] (Body -> World at the end/center of window)
                # 注意：PoseNet 的输入依然是 Raw IMU (xb)，因为它需要重力向量来校准水平面
                R_pred = pose_net(xb) 

            # --- Step 3: 坐标系对齐 (Coordinate Alignment) ---
            # 将 Linear Acc 旋转到 "PoseNet 预测的世界系"
            # xb_aligned: [B, T, 6]
            xb_aligned = rotate_imu(xb_linear, R_pred)
            
            # --- Step 4: 制作训练标签 (Label Engineering) ---
            # 核心思想：把真值位移 (World) 逆向旋转回 PoseNet 的局部系
            # 这样 Navigator 只需要学习 "在 PoseNet 姿态基础上的修正量"
            
            # 构造 3D 位移向量 (确保维度正确)
            # yb_dp 是 [B, 3]
            dp_world = yb_dp.unsqueeze(-1) # [B, 3, 1]
            
            # 逆旋转: R_pred^T @ dp_world
            # 结果 dp_local 代表：在 PoseNet 认为的"前方"坐标系下，人实际走了多少
            dp_local = torch.matmul(R_pred.transpose(1, 2), dp_world).squeeze(-1) # [B, 3]
            
            # 提取水平面目标 (XY)
            target_dx = dp_local[:, 0]
            target_dy = dp_local[:, 1]
            
            # 转换为极坐标标签
            target_speed = torch.sqrt(target_dx**2 + target_dy**2 + 1e-8).unsqueeze(1) # [B, 1]
            target_cos = (target_dx / target_speed) # [B, 1]
            target_sin = (target_dy / target_speed) # [B, 1]

            # --- Step 5: 网络前向与 Loss ---
            # Navigator 输入是对齐后的 Linear IMU
            pred_out = navigator(xb_aligned) # [B, 3] -> [cos, sin, speed]
            
            pred_cos = pred_out[:, 0:1]
            pred_sin = pred_out[:, 1:2]
            pred_speed = pred_out[:, 2:3]
            
            # Loss 设计
            # 1. 方向 Loss (MSE on Cos/Sin)
            loss_cos = F.mse_loss(pred_cos, target_cos)
            loss_sin = F.mse_loss(pred_sin, target_sin)
            
            # 2. 速度 Loss (MSE on Speed)
            loss_speed = F.mse_loss(pred_speed, target_speed)
            
            # 总 Loss
            loss = loss_cos + loss_sin + loss_speed
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(navigator.parameters(), 1.0)
            optimizer.step()
            
            train_cnt += xb.size(0)
            total_train_loss += loss.item() * xb.size(0)

        # --- Validation Loop (Similar logic) ---
        avg_train_loss = total_train_loss / max(train_cnt, 1)
        
        navigator.eval()
        total_val_loss = 0.0
        val_cnt = 0
        
        with torch.no_grad():
            for batch in val_loader:
                xb, yb_dp, yb_ori = batch
                
                acc_raw = xb[:, :, 3:6]
                gyro_raw = xb[:, :, 0:3]
                acc_linear = remove_gravity(acc_raw, yb_ori)
                xb_linear = torch.cat([gyro_raw, acc_linear], dim=2)
                
                R_pred = pose_net(xb)
                xb_aligned = rotate_imu(xb_linear, R_pred)
                
                dp_world = yb_dp.unsqueeze(-1)
                dp_local = torch.matmul(R_pred.transpose(1, 2), dp_world).squeeze(-1)
                
                t_dx, t_dy = dp_local[:, 0], dp_local[:, 1]
                t_spd = torch.sqrt(t_dx**2 + t_dy**2 + 1e-8).unsqueeze(1)
                t_cos, t_sin = t_dx/t_spd, t_dy/t_spd
                
                pred = navigator(xb_aligned)
                v_loss = F.mse_loss(pred[:,0:1], t_cos) + \
                         F.mse_loss(pred[:,1:2], t_sin) + \
                         F.mse_loss(pred[:,2:3], t_spd)
                         
                total_val_loss += v_loss.item() * xb.size(0)
                val_cnt += xb.size(0)
                
        avg_val_loss = total_val_loss / max(val_cnt, 1)
        
        # Save Best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(navigator.state_dict(), ckpt)
            
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[Nav Ep {ep+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# ======= 主流程 =======

def main():
    print("=" * 60)
    print("NavCorrector: PoseNet + Polar Navigator (with Gravity Removal)")
    print("=" * 60)

    # 1. 加载数据
    # 注意：我们需要 return_ori=True 以便计算重力
    # yb_rel 用于 PoseNet 训练，yb_dp 用于 Navigator 训练
    print("\n📊 加载训练数据...")
    (x_tr, _ylen_tr, _yhead_tr, ydp_tr, yori_tr, yrel_tr,
     x_va, _ylen_va, _yhead_va, ydp_va, yori_va, yrel_va) = load_data_2d_ridi_absheading(
        ridi_root, device, window_size, stride,
        return_ori=True,        # 必须: 用于去重力
        return_rel_ori=True,    # 必须: 用于 PoseNet 训练
        return_delta_p=True,    # 必须: 用于 Navigator 训练
        use_abs_heading=False
    )
    
    # PoseNet 数据集: (Input, Relative Rotation Label)
    pose_dataset = TensorDataset(x_tr, yrel_tr)
    pose_loader = DataLoader(pose_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Navigator 数据集: (Input, Global Displacement, Orientation for Gravity)
    nav_train_ds = TensorDataset(x_tr, ydp_tr, yori_tr)
    nav_val_ds = TensorDataset(x_va, ydp_va, yori_va)
    
    nav_train_loader = DataLoader(nav_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    nav_val_loader = DataLoader(nav_val_ds, batch_size=batch_size, shuffle=False)

    # 2. 初始化模型
    in_ch = x_tr.shape[-1]
    pose_net = PoseNetTransformer(imu_dim=6, d_model=128).to(device)
    # Navigator 输入维度还是 6 (Gyro + Linear Acc)，输出维度改为了 3
    navigator = Navigator(imu_dim=in_ch, feat_dim=feat_dim).to(device)

    # 3. 训练 PoseNet
    train_pose_net(pose_net, pose_loader)

    # 4. 训练 Navigator (Corrected)
    train_navigator_corrected(pose_net, navigator, nav_train_loader, nav_val_loader)

    print("\n✅ 训练完成")

if __name__ == "__main__":
    main()