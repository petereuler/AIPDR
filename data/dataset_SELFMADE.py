import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import os
from scipy.io import loadmat


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def moving_average(x, k):
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


def yaw_from_positions(pos_xy, min_displacement=0.05, smooth_window=5):
    """
    从位置计算 Course Angle，包含平滑和去噪
    """
    n = len(pos_xy)
    if n < 2:
        return np.zeros(n)
        
    # 1. 强力平滑
    if smooth_window > 1:
        # 使用多次平滑以抑制高频噪声
        pos_smooth = moving_average(pos_xy, smooth_window)
        pos_smooth = moving_average(pos_smooth, smooth_window)
    else:
        pos_smooth = pos_xy
        
    # 2. 计算速度向量
    vel = np.gradient(pos_smooth, axis=0)
    
    # 3. 计算航向角，过滤微小位移
    yaws = np.zeros(n)
    
    # 初始角度
    if np.linalg.norm(vel[0]) > min_displacement:
        yaws[0] = np.arctan2(vel[0, 1], vel[0, 0])
        
    for i in range(1, n):
        v = vel[i]
        mag = np.linalg.norm(v)
        
        if mag > min_displacement:
            # 只有当速度足够大时才更新航向
            current_yaw = np.arctan2(v[1], v[0])
            yaws[i] = current_yaw
        else:
            # 否则保持上一个时刻的航向
            yaws[i] = yaws[i-1]
            
    # 4. 再次平滑角度 (在 sin/cos 域)
    # 这能消除某些跳变
    cos_yaw = np.cos(yaws)
    sin_yaw = np.sin(yaws)
    cos_smooth = moving_average(cos_yaw, smooth_window)
    sin_smooth = moving_average(sin_yaw, smooth_window)
    yaws_smooth = np.arctan2(sin_smooth, cos_smooth)
            
    return yaws_smooth


def load_selfmade_raw(imu_or_all_path, gt_path=None, crop_head=0, crop_tail=0):
    ext = os.path.splitext(imu_or_all_path)[1].lower()
    if ext == '.mat':
        mat = loadmat(imu_or_all_path)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        arr = None
        for k in keys:
            v = mat[k]
            if isinstance(v, np.ndarray) and v.ndim >= 2 and v.dtype.kind in ('f', 'i'):
                arr = v
                break
        if arr is None:
            raise RuntimeError('No numeric array found in MAT file')
        a = np.squeeze(arr)
        if a.ndim != 2:
            a = a.reshape(a.shape[0], -1)
        if a.shape[0] == 11:
            channels = a
        elif a.shape[1] == 11:
            channels = a.T
        elif a.shape[0] > 11:
            channels = a[:11, :]
        elif a.shape[1] > 11:
            channels = a[:, :11].T
        else:
            raise RuntimeError('MAT array does not contain 11 channels')
        gyro = channels[0:3, :].T
        acc = channels[3:6, :].T
        heading_deg = channels[6, :]
        x = channels[9, :]
        y = channels[10, :]
    else:
        df_imu = pd.read_csv(imu_or_all_path)
        df_gt = None
        if gt_path is not None and gt_path != imu_or_all_path:
            df_gt = pd.read_csv(gt_path)
        if df_gt is None:
            values = df_imu.values
            gyro = values[:, 0:3]
            acc = values[:, 3:6]
            heading_deg = values[:, 6]
            x = values[:, 9]
            y = values[:, 10]
        else:
            v_imu = df_imu.values
            v_gt = df_gt.values
            gyro = v_imu[:, 0:3]
            acc = v_imu[:, 3:6]
            heading_deg = v_gt[:, 0]
            x = v_gt[:, 1]
            y = v_gt[:, 2]
    if crop_head or crop_tail:
        h = int(max(0, crop_head))
        t = int(max(0, crop_tail))
        if h > 0:
            gyro = gyro[h:]
            acc = acc[h:]
            heading_deg = heading_deg[h:]
            x = x[h:]
            y = y[h:]
        if t > 0:
            gyro = gyro[:-t]
            acc = acc[:-t]
            heading_deg = heading_deg[:-t]
            x = x[:-t]
            y = y[:-t]
    n = min(len(gyro), len(acc), len(heading_deg), len(x), len(y))
    gyro = gyro[:n]
    acc = acc[:n]
    heading_rad = np.radians(heading_deg[:n])
    pos_xyz = np.stack([x[:n], y[:n], np.zeros(n, dtype=float)], axis=1)
    ori_stub = heading_rad.reshape(-1, 1)
    
    # # SELFMADE数据集降采样2（每隔一个样本取一个）
    # gyro = gyro[::2]
    # acc = acc[::2]
    # pos_xyz = pos_xyz[::2]
    # ori_stub = ori_stub[::2]
    
    return gyro, acc, pos_xyz, ori_stub


def window_dataset(gyro_data, acc_data, pos_data, ori_data, window_size=160, stride=36, filter_window=5, smooth_heading=True, heading_sigma=3, smooth_length=False, length_sigma=1.0):
    m = min(gyro_data.shape[0], acc_data.shape[0], pos_data.shape[0], ori_data.shape[0])
    gyro_data = gyro_data[:m]
    acc_data = acc_data[:m]
    pos_data = pos_data[:m]
    ori_data = ori_data[:m]

    pos_xyz = pos_data
    if filter_window and filter_window > 1:
        pos_xyz = moving_average(pos_xyz, filter_window)
    pos_xy = pos_xyz[:, :2]

    x_gyro = []
    x_acc = []
    y_len = []
    y_head = []

    start_0 = window_size // 2 - stride // 2
    end_0 = window_size // 2 + stride // 2
    start_0 = max(0, min(start_0, len(pos_xy) - 1))
    end_0 = max(0, min(end_0, len(pos_xy) - 1))
    init_pos = pos_xy[start_0, :]

    diff_0 = pos_xy[end_0] - pos_xy[start_0]
    init_head = float(np.arctan2(diff_0[1], diff_0[0]))

    max_start = gyro_data.shape[0] - window_size - 1
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
            if i == 0:
                curr_chord_angle = 0.0
            else:
                curr_chord_angle = prev_chord_angle
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
                if np.linalg.norm(prev_diff) < 1e-6:
                    prev_chord_angle = curr_chord_angle
                else:
                    prev_chord_angle = np.arctan2(prev_diff[1], prev_diff[0])
                delta_head = wrap_angle(curr_chord_angle - prev_chord_angle)

        y_len.append(np.array([delta_len], dtype=np.float32))
        y_head.append(np.array([delta_head], dtype=np.float32))

    x_gyro = np.array(x_gyro)
    x_acc = np.array(x_acc)
    y_len = np.array(y_len)
    y_head = np.array(y_head)

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

    return [x_gyro, x_acc], [y_len, y_head], init_pos, init_head
