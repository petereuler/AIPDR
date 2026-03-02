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
    加载 OxIOD 原始数据：IMU(gyro/acc) 与 GT 位置/姿态（XYZ）。

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

def window_dataset(gyro_data, acc_data, pos_data, ori_data, window_size=160, stride=36, filter_window=20, smooth_heading=True, heading_sigma=5, smooth_length=False, length_sigma=5, return_rel_ori=False, return_delta_p=False):
    pos_xyz = pos_data
    if filter_window and filter_window > 1:
        pos_xyz = moving_average(pos_xyz, filter_window)
    pos_xy = pos_xyz[:, :2]

    # 使用弦角作为航向参考，不需要预先计算 yaw 序列

    x_gyro = []
    x_acc = []
    y_len = []
    y_head = []
    y_rel = []
    y_dp = []

    # 初始化起点与初始航向
    start_0 = window_size // 2 - stride // 2
    end_0 = window_size // 2 + stride // 2
    start_0 = max(0, min(start_0, len(pos_xy) - 1))
    end_0 = max(0, min(end_0, len(pos_xy) - 1))

    init_pos = pos_xy[start_0, :]
    diff_0 = pos_xy[end_0] - pos_xy[start_0]
    init_head = float(np.arctan2(diff_0[1], diff_0[0]))

    max_start = gyro_data.shape[0] - window_size - 1
    prev_chord_angle = 0.0
    for i, idx in enumerate(range(0, max_start, stride)):
        xg = gyro_data[idx + 1: idx + 1 + window_size, :]
        xa = acc_data[idx + 1: idx + 1 + window_size, :]

        x_gyro.append(xg)
        x_acc.append(xa)

        start_idx = idx + window_size // 2 - stride // 2
        end_idx = idx + window_size // 2 + stride // 2

        start_idx = max(0, min(start_idx, len(pos_xy) - 1))
        end_idx = max(0, min(end_idx, len(pos_xy) - 1))

        pos_start_xy = pos_xy[start_idx, :]
        pos_end_xy = pos_xy[end_idx, :]
        pos_start_xyz = pos_xyz[start_idx, :]
        pos_end_xyz = pos_xyz[end_idx, :]

        delta_len = np.linalg.norm(pos_end_xyz - pos_start_xyz)
        curr_diff = pos_end_xy - pos_start_xy
        if np.linalg.norm(curr_diff) < 1e-6:
            curr_chord_angle = 0.0 if i == 0 else prev_chord_angle
        else:
            curr_chord_angle = np.arctan2(curr_diff[1], curr_diff[0])

        if i == 0:
            delta_head = 0.0
            prev_chord_angle = curr_chord_angle
        else:
            prev_start = start_idx - stride
            if prev_start < 0:
                delta_head = 0.0
                prev_chord_angle = curr_chord_angle
            else:
                prev_p = pos_xy[prev_start]
                prev_diff = pos_start_xy - prev_p
                prev_chord_angle = np.arctan2(prev_diff[1], prev_diff[0])
                delta_head = wrap_angle(curr_chord_angle - prev_chord_angle)

        y_len.append(np.array([delta_len], dtype=np.float32))
        y_head.append(np.array([delta_head], dtype=np.float32))
        if return_delta_p:
            y_dp.append((pos_end_xyz - pos_start_xyz).astype(np.float32))
        if return_rel_ori:
            q_start = ori_data[start_idx].astype(np.float32)
            q_end = ori_data[end_idx].astype(np.float32)
            y_rel.append(quat_mul(q_end, quat_conj(q_start)))

    x_gyro = np.array(x_gyro)
    x_acc = np.array(x_acc)
    y_len = np.array(y_len)
    y_head = np.array(y_head)
    if return_rel_ori:
        y_rel = np.array(y_rel)
    if return_delta_p:
        y_dp = np.array(y_dp)

    # 在平滑之前进行数据清洗：基于步长判断静止状态
    if len(y_len) > 0 and len(y_head) > 0:
        stationary_mask = np.abs(y_len.flatten()) < 0.01
        y_len[stationary_mask, 0] = 0.0
        y_head[stationary_mask, 0] = 0.0

    if smooth_length and len(y_len) > 0:
        y_len_smooth = gaussian_filter1d(y_len.flatten(), sigma=length_sigma)
        y_len = y_len_smooth.reshape(-1, 1)

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
