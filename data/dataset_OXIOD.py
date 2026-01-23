import numpy as np
import pandas as pd
import quaternion
from scipy.ndimage import gaussian_filter1d


def wrap_angle(angle):
    """将角度归一化到 [-pi, pi] 范围，支持标量或数组。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=np.float32)

def moving_average(x, k):
    """简易滑动平均滤波，窗口 k>=1；k<=1 时原样返回。"""
    if k is None or k <= 1:
        return x
    k = int(k)
    if k <= 1:
        return x
    kernel = np.ones(k, dtype=float) / float(k)
    if isinstance(x, np.ndarray) and x.ndim == 1:
        return np.convolve(x, kernel, mode='same')
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return np.stack([np.convolve(x[:, i], kernel, mode='same') for i in range(x.shape[1])], axis=1)
    return x 

def quaternion_to_euler(q):
    """
    将四元数转换为欧拉角 (roll, pitch, yaw)
    参数:
    - q (np.ndarray or quaternion.quaternion): 四元数，形状为 (4,)
    
    返回:
    - euler (np.ndarray): 对应的欧拉角 [roll, pitch, yaw]，单位是弧度
    """
    q = quaternion.from_float_array(q) if isinstance(q, np.ndarray) else q
    rotation_matrix = quaternion.as_rotation_matrix(q)
    
    # 从旋转矩阵提取欧拉角
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # roll (旋转绕X轴)
    pitch = np.arcsin(-rotation_matrix[2, 0])  # pitch (旋转绕Y轴)
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # yaw (旋转绕Z轴)
    
    return np.array([roll, pitch, yaw])

def yaw_from_quaternion_array(ori_array):
    yaws = []
    for q in ori_array:
        e = quaternion_to_euler(q)
        yaws.append(e[2])
    return np.array(yaws)

def load_oxiod_raw(imu_data_filename, gt_data_filename):
    """
    加载 OxIOD 原始数据：IMU(gyro/acc) 与 GT 位置/姿态（3D）。

    参数:
    - imu_data_filename: IMU 数据的文件路径
    - gt_data_filename: 地面真实数据的文件路径

    返回:
    - gyro_data: 陀螺仪数据 (N, 3)
    - acc_data: 加速度数据 (N, 3)
    - pos_data: 位置数据 (N, 3)
    - ori_data: 姿态（四元数 [w, x, y, z]）(N, 4)
    """
    # 去除表头，防止训练中epoch第一轮读取表头
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    # 对数据进行切片以去除开头和结尾的无效数据
    imu_data = imu_data[1200:-300]
    gt_data = gt_data[1200:-300]

    gyro_data = imu_data[:, 4:7]
    acc_data = imu_data[:, 10:13]

    pos_data = gt_data[:, 2:5]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)  # 得到四元数顺序：[w, x, y, z]

    return gyro_data, acc_data, pos_data, ori_data

def window_dataset(gyro_data, acc_data, pos_data, ori_data, mode = "2d", window_size = 160, stride = 36, filter_window = 20, smooth_heading = True, heading_sigma = 5, smooth_length = False, length_sigma = 5, return_rel_ori=False, return_delta_p=False):
    mid = window_size // 2 - stride // 2
    if mode == "2d":
        pos2d = pos_data[:, :2]
        if filter_window and filter_window > 1:
            pos2d = moving_average(pos2d, filter_window)

        # [修改] 使用 Chord Angle (弦角) 而非 Instantaneous Course (切线角)
        # 不需要预先计算 yaw_series

        x_gyro = []
        x_acc = []
        y_len = []
        y_head = []
        y_rel = []
        y_dp = []
        
        # 初始化
        # 为了计算 chord_angle_prev，我们需要知道前一个窗口的弦角
        # 但是窗口是独立的样本。
        # 这里的关键是：我们定义的 y_head (dh) 是 "当前步的弦角" - "前一步的弦角"
        # 因此我们需要在遍历时维护状态，或者根据索引回溯。
        
        # 由于我们是按照 stride 遍历的，idx 对应的是 step start。
        # idx 序列: 0, stride, 2*stride, ...
        # step k: P[idx + mid - stride/2] -> P[idx + mid + stride/2]
        # let center_idx = idx + mid
        # step k: P[center_idx - stride/2] -> P[center_idx + stride/2]
        # start_k = center_idx - stride/2
        # end_k   = center_idx + stride/2
        # chord_k = atan2(P[end_k] - P[start_k])
        
        # prev step (k-1): start_prev = start_k - stride, end_prev = end_k - stride = start_k
        # chord_prev = atan2(P[end_prev] - P[start_prev]) = atan2(P[start_k] - P[start_k - stride])
        
        # 因此，对于每个 idx，我们可以直接计算当前弦角和前一个弦角。
        
        # 计算初始位置和航向 (用于重建)
        # 重建时，初始状态是 P0 和 H0.
        # 第一步预测 L0, dH0. 更新: H1 = H0 + dH0. P1 = P0 + L0 * u(H1).
        # 我们希望 P1 落在真值上，所以 H1 必须是 chord_0 的角度。
        # 即 H0 + dH0 = chord_0.
        # 如果我们设 H0 = chord_0 (即 init_head 为第一步的弦角)，则 dH0 应该为 0。
        # 随后的步骤 dH_k = chord_k - chord_{k-1}.
        
        # init_pos 取第一个窗口的起点 (a)
        idx_0 = 0
        a_0 = idx_0 + window_size // 2 - stride // 2
        b_0 = idx_0 + window_size // 2 + stride // 2
        # 确保索引安全
        a_0 = max(0, min(a_0, len(pos2d)-1))
        b_0 = max(0, min(b_0, len(pos2d)-1))
        
        init_pos = pos2d[a_0, :]
        
        # 计算第一个 chord angle 作为 init_head
        diff_0 = pos2d[b_0] - pos2d[a_0]
        init_head = float(np.arctan2(diff_0[1], diff_0[0]))

        max_start = gyro_data.shape[0] - window_size - 1
        for i, idx in enumerate(range(0, max_start, stride)):
            xg = gyro_data[idx + 1: idx + 1 + window_size, :]
            xa = acc_data [idx + 1: idx + 1 + window_size, :]
            
            x_gyro.append(xg)
            x_acc .append(xa)

            a = idx + window_size // 2 - stride // 2
            b = idx + window_size // 2 + stride // 2
            
            # 索引边界保护
            a = max(0, min(a, len(pos2d)-1))
            b = max(0, min(b, len(pos2d)-1))
            
            pa = pos2d[a, :]
            pb = pos2d[b, :]
            
            # 1. 步长 (弦长)
            delta_len = np.linalg.norm(pb - pa)

            # 2. 当前弦角
            curr_diff = pb - pa
            # 处理静止情况，防止 NaN (虽然 OXIOD 数据通常在动)
            if np.linalg.norm(curr_diff) < 1e-6:
                curr_chord_angle = 0.0 if i == 0 else prev_chord_angle # 保持不变
            else:
                curr_chord_angle = np.arctan2(curr_diff[1], curr_diff[0])

            # 3. 航向变化量 (dh)
            if i == 0:
                # 第一步，如果 init_head = curr_chord_angle，则 dh = 0
                delta_head = 0.0
            else:
                # 计算前一步的弦角
                # 前一步是从 a-stride 到 a (即 current a 是上一布的 b)
                prev_a = a - stride
                if prev_a < 0:
                    # 理论上不应发生，除非 stride > a
                    # fallback
                    delta_head = 0.0
                else:
                    prev_p = pos2d[prev_a]
                    prev_diff = pa - prev_p
                    if np.linalg.norm(prev_diff) < 1e-6:
                        prev_chord_angle = curr_chord_angle
                    else:
                        prev_chord_angle = np.arctan2(prev_diff[1], prev_diff[0])
                    
                    delta_head = wrap_angle(curr_chord_angle - prev_chord_angle)

            y_len .append(np.array([delta_len], dtype=np.float32))
            y_head.append(np.array([delta_head], dtype=np.float32))
            if return_delta_p:
                pa3 = pos_data[a, :]
                pb3 = pos_data[b, :]
                y_dp.append((pb3 - pa3).astype(np.float32))
            if return_rel_ori:
                qa = ori_data[a].astype(np.float32)
                qb = ori_data[b].astype(np.float32)
                y_rel.append(quat_mul(qb, quat_conj(qa)))

        x_gyro = np.array(x_gyro)
        x_acc  = np.array(x_acc)
        y_len  = np.array(y_len)
        y_head = np.array(y_head)
        if return_rel_ori:
            y_rel = np.array(y_rel)
        if return_delta_p:
            y_dp = np.array(y_dp)
        
        # 在平滑之前进行数据清洗：基于步长判断静止状态
        # 如果步长绝对值小于阈值，说明处于静止状态，将步长和航向角都设为0
        if len(y_len) > 0 and len(y_head) > 0:
            # 基于步长判断是否静止
            stationary_mask = np.abs(y_len.flatten()) < 0.01  # 步长小于1cm认为静止

            # 将静止状态的样本标签设为0
            y_len[stationary_mask, 0] = 0.0
            y_head[stationary_mask, 0] = 0.0

        # 对步长进行平滑处理（提高真值轨迹的光滑性）
        if smooth_length and len(y_len) > 0:
            y_len_smooth = gaussian_filter1d(y_len.flatten(), sigma=length_sigma)
            y_len = y_len_smooth.reshape(-1, 1)
        
        # 对航向角进行平滑处理（提高真值轨迹的光滑性）
        if smooth_heading and len(y_head) > 0:
            y_head_smooth = gaussian_filter1d(y_head.flatten(), sigma=heading_sigma)
            y_head = y_head_smooth.reshape(-1, 1)

        if return_rel_ori and return_delta_p:
            return [x_gyro, x_acc], [y_len, y_head, y_rel, y_dp], init_pos, init_head
        if return_rel_ori:
            return [x_gyro, x_acc], [y_len, y_head, y_rel], init_pos, init_head
        if return_delta_p:
            return [x_gyro, x_acc], [y_len, y_head, y_dp], init_pos, init_head
        return [x_gyro, x_acc], [y_len, y_head], init_pos, init_head

    elif mode == "3d":
        mid = window_size // 2 - stride // 2
        init_pos = pos_data[mid, :]
        init_euler = quaternion_to_euler(ori_data[mid, :])

        x_gyro = []
        x_acc = []
        y_delta_p = []
        y_delta_euler = []

        max_start = gyro_data.shape[0] - window_size - 1
        for idx in range(0, max_start, stride):
            xg = gyro_data[idx + 1: idx + 1 + window_size, :]
            xa = acc_data [idx + 1: idx + 1 + window_size, :]
            
            x_gyro.append(xg)
            x_acc .append(xa)

            a = idx + window_size // 2 - stride // 2
            b = idx + window_size // 2 + stride // 2

            p_a = pos_data[a, :]
            p_b = pos_data[b, :]
            q_a = quaternion.from_float_array(ori_data[a, :])
            q_b = quaternion.from_float_array(ori_data[b, :])

            rotation_matrix = quaternion.as_rotation_matrix(q_a)
            delta_p = rotation_matrix.T @ (p_b - p_a)

            e_a = quaternion_to_euler(q_a)
            e_b = quaternion_to_euler(q_b)
            delta_euler = wrap_angle(e_b - e_a)

            y_delta_p.append(delta_p)
            y_delta_euler.append(delta_euler)

        x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
        x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
        y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
        y_delta_euler = np.reshape(y_delta_euler, (len(y_delta_euler), y_delta_euler[0].shape[0]))

        return [x_gyro, x_acc], [y_delta_p, y_delta_euler], init_pos, init_euler

    else:
        raise ValueError("mode must be '2d' or '3d'")
