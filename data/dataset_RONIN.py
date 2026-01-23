import os
import json
import h5py
import numpy as np
import quaternion
from scipy.ndimage import gaussian_filter1d
from RONIN.source.math_util import orientation_to_angles


def wrap_angle(angle):
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


def _load_sequence(seq_path):
    with open(os.path.join(seq_path, 'info.json')) as f:
        info = json.load(f)
    with h5py.File(os.path.join(seq_path, 'data.hdf5')) as f:
        ts = np.copy(f['synced/time'])
        gyro_uncalib = np.copy(f['synced/gyro_uncalib'])
        acce_uncalib = np.copy(f['synced/acce'])
        tango_pos = np.copy(f['pose/tango_pos'])
        if 'pose/tango_ori' in f.keys():
            init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])
        else:
            init_tango_ori = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    gyro = gyro_uncalib - np.array(info['imu_init_gyro_bias'])
    acce = np.array(info['imu_acce_scale']) * (acce_uncalib - np.array(info['imu_acce_bias']))
    ori_src = info.get('ori_source', 'game_rv')
    with h5py.File(os.path.join(seq_path, 'data.hdf5')) as f:
        if ori_src == 'game_rv' and 'synced/game_rv' in f.keys():
            ori = np.copy(f['synced/game_rv'])
        elif 'synced/rv' in f.keys():
            ori = np.copy(f['synced/rv'])
        else:
            ori = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (ts.shape[0], 1))
    ori_q = quaternion.from_float_array(ori)
    rot_imu_to_tango = quaternion.quaternion(*info.get('start_calibration', [1.0, 0.0, 0.0, 0.0]))
    init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
    ori_q = init_rotor * ori_q
    nz = np.zeros((gyro.shape[0], 1))
    gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
    acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))
    glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
    glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]
    dt = (ts[1:] - ts[:-1])[:, None]
    glob_v = (tango_pos[1:] - tango_pos[:-1]) / dt
    ts = ts[1:]
    return ts, np.concatenate([glob_gyro[1:], glob_acce[1:]], axis=1), glob_v[:, :2], quaternion.as_float_array(ori_q)[1:], tango_pos[1:]


def load_ronin_raw(seq_path):
    ts, feat, vel2, ori, pos = _load_sequence(seq_path)
    gyro = feat[:, :3]
    acc = feat[:, 3:6]
    pos3 = pos
    angles = orientation_to_angles(ori)
    yaw = angles[:, 0]
    
    # RONIN数据集降采样2（每隔一个样本取一个）
    # gyro = gyro[::2]
    # acc = acc[::2]
    # pos3 = pos3[::2]
    # yaw = yaw[::2]
    
    return gyro, acc, pos3, yaw.reshape(-1, 1)


def window_dataset(gyro_data, acc_data, pos_data, ori_data, mode="2d", window_size=200, stride=10, filter_window=10, smooth_heading=True, heading_sigma=1, smooth_length=False, length_sigma=1.0, return_rel_ori=False, return_delta_p=False):
    mid = window_size // 2 - stride // 2
    m = min(gyro_data.shape[0], acc_data.shape[0], pos_data.shape[0], ori_data.shape[0])
    gyro_data = gyro_data[:m]
    acc_data = acc_data[:m]
    pos_data = pos_data[:m]
    ori_data = ori_data[:m]
    if mode == "2d":
        pos2d = pos_data[:, :2]
        if filter_window and filter_window > 1:
            pos2d = moving_average(pos2d, filter_window)
        
        x_gyro = []
        x_acc = []
        y_len = []
        y_head = []
        y_rel = []
        y_dp = []
        
        # init_pos & init_head
        idx_0 = 0
        a_0 = idx_0 + window_size // 2 - stride // 2
        b_0 = idx_0 + window_size // 2 + stride // 2
        a_0 = max(0, min(a_0, len(pos2d)-1))
        b_0 = max(0, min(b_0, len(pos2d)-1))
        init_pos = pos2d[a_0, :]
        
        diff_0 = pos2d[b_0] - pos2d[a_0]
        init_head = float(np.arctan2(diff_0[1], diff_0[0]))

        max_start = gyro_data.shape[0] - window_size - 1
        for i, idx in enumerate(range(0, max_start, stride)):
            xg = gyro_data[idx + 1: idx + 1 + window_size, :]
            xa = acc_data[idx + 1: idx + 1 + window_size, :]
            x_gyro.append(xg)
            x_acc.append(xa)
            
            a = idx + window_size // 2 - stride // 2
            b = idx + window_size // 2 + stride // 2
            a = max(0, min(a, len(pos2d)-1))
            b = max(0, min(b, len(pos2d)-1))
            
            pa = pos2d[a, :]
            pb = pos2d[b, :]
            
            delta_len = np.linalg.norm(pb - pa)
            
            curr_diff = pb - pa
            if np.linalg.norm(curr_diff) < 1e-6:
                curr_chord_angle = 0.0 if i == 0 else prev_chord_angle
            else:
                curr_chord_angle = np.arctan2(curr_diff[1], curr_diff[0])
            
            if i == 0:
                delta_head = 0.0
            else:
                prev_a = a - stride
                if prev_a < 0:
                    delta_head = 0.0
                else:
                    prev_p = pos2d[prev_a]
                    prev_diff = pa - prev_p
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

        # # 在平滑之前进行数据清洗：基于步长判断静止状态
        # # 如果步长绝对值小于阈值，说明处于静止状态，将步长和航向角都设为0
        # if len(y_len) > 0 and len(y_head) > 0:
        #     # 基于步长判断是否静止
        #     stationary_mask = np.abs(y_len.flatten()) < 0.01  # 步长小于1cm认为静止

        #     # 将静止状态的样本标签设为0
        #     y_len[stationary_mask, 0] = 0.0
        #     y_head[stationary_mask, 0] = 0.0

        # 对步长进行平滑处理（提高真值轨迹的光滑性）
        if smooth_length and len(y_len) > 0:
            y_len_smooth = gaussian_filter1d(y_len.flatten(), sigma=length_sigma)
            y_len = y_len_smooth.reshape(-1, 1)
        
        # 对航向角进行平滑处理（提高真值轨迹的光滑性）
        if smooth_heading and len(y_head) > 0:
            y_head_smooth = gaussian_filter1d(y_head.flatten(), sigma=heading_sigma)
            y_head = y_head_smooth.reshape(-1, 1)
        
        return [x_gyro, x_acc], [y_len, y_head], init_pos, init_head
    elif mode == "3d":
        raise ValueError("RONIN helper only provides 2d windows here")
    else:
        raise ValueError("mode must be '2d' or '3d'")
