import os
import numpy as np
from scipy.ndimage import gaussian_filter1d


def wrap_angle(angle):
    """将角度归一化到 [-pi, pi] 范围，支持标量或数组。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def yaw_from_quat(q):
    q = np.array(q, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, 4)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_conj(q):
    q = np.array(q, dtype=np.float32)
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


def _load_txt(path):
    with open(path, 'r') as f:
        lines = [l for l in f if l.strip() and not l.startswith('#')]
    return np.loadtxt(lines)


def _interp_to(t_src, data, t_tgt):
    if t_src.ndim != 1:
        t_src = t_src.reshape(-1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    t_tgt = np.clip(t_tgt, t_src[0], t_src[-1])
    out = np.zeros((len(t_tgt), data.shape[1]), dtype=np.float32)
    for i in range(data.shape[1]):
        out[:, i] = np.interp(t_tgt, t_src, data[:, i])
    return out


ORI_ORDER = "wxyz"  # RIDI orientation.txt uses w, x, y, z
ORI_IS_W2B = False  # RIDI orientation is body->world
POSE_ORDER = "xyzw"  # RIDI pose.txt quaternions are x, y, z, w (Tango), swap to wxyz


def load_ridi_raw(seq_dir, acc_source="acce"):
    """
    加载 RIDI 原始数据：IMU(gyro/acc) 与 GT 位置/姿态。

    Args:
        seq_dir: RIDI 单条序列目录

    Returns:
        gyro_data: (N, 3)
        acc_data: (N, 3)
        pos_data: (N, 3)
        ori_data: (N, 4)
    """
    gyro_path = os.path.join(seq_dir, "gyro.txt")
    if acc_source == "acce":
        acc_path = os.path.join(seq_dir, "acce.txt")
    elif acc_source == "linacce":
        acc_path = os.path.join(seq_dir, "linacce.txt")
    else:
        raise ValueError(f"Unknown acc_source: {acc_source}")
    pose_path = os.path.join(seq_dir, "pose.txt")
    ori_path = os.path.join(seq_dir, "orientation.txt")

    gyro_arr = _load_txt(gyro_path)
    acc_arr = _load_txt(acc_path)
    pose_arr = _load_txt(pose_path)
    ori_arr = _load_txt(ori_path) if os.path.exists(ori_path) else None

    t_gyro = gyro_arr[:, 0]
    gyro = gyro_arr[:, 1:4]

    t_acc = acc_arr[:, 0]
    acc = _interp_to(t_acc, acc_arr[:, 1:4], t_gyro)

    t_pose = pose_arr[:, 0]
    pos = _interp_to(t_pose, pose_arr[:, 1:4], t_gyro)

    if pose_arr is not None and pose_arr.shape[1] >= 8:
        t_pose = pose_arr[:, 0]
        pose_quat = pose_arr[:, -4:]
        if POSE_ORDER == "xyzw":
            pose_quat = pose_quat[:, [3, 0, 1, 2]]
        ori = _interp_to(t_pose, pose_quat, t_gyro)
        # normalize quaternion to avoid drift from interpolation
        norm = np.linalg.norm(ori, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-8, None)
        ori = ori / norm
    elif ori_arr is not None and ori_arr.shape[1] >= 5:
        t_ori = ori_arr[:, 0]
        ori = _interp_to(t_ori, ori_arr[:, 1:5], t_gyro)
        if ORI_ORDER == "xyzw":
            ori = ori[:, [3, 0, 1, 2]]
        if ORI_IS_W2B:
            ori = np.stack([ori[:, 0], -ori[:, 1], -ori[:, 2], -ori[:, 3]], axis=1)
    else:
        ori = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(t_gyro), 1))

    return gyro.astype(np.float32), acc.astype(np.float32), pos.astype(np.float32), ori.astype(np.float32)


def window_dataset(gyro_data, acc_data, pos_data, ori_data, mode="2d",
                   window_size=160, stride=36, filter_window=20,
                   smooth_heading=True, heading_sigma=1.5,
                   smooth_length=False, length_sigma=1.0,
                   return_abs_heading=False, return_ori=False, return_rel_ori=False,
                   return_delta_p=False,
                   abs_heading_from_ori=False, align_heading_to_init_pose=False):
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
        y_abs = []
        y_ori = []
        y_rel = []
        y_dp = []

        idx_0 = 0
        a_0 = idx_0 + window_size // 2 - stride // 2
        b_0 = idx_0 + window_size // 2 + stride // 2
        a_0 = max(0, min(a_0, len(pos2d)-1))
        b_0 = max(0, min(b_0, len(pos2d)-1))

        init_pos = pos2d[a_0, :]
        if abs_heading_from_ori:
            yaw0 = float(yaw_from_quat(ori_data[b_0])[0])
            init_head = 0.0
        else:
            diff_0 = pos2d[b_0] - pos2d[a_0]
            init_head = float(np.arctan2(diff_0[1], diff_0[0]))
            if align_heading_to_init_pose:
                yaw0 = float(yaw_from_quat(ori_data[b_0])[0])
                init_head = 0.0

        max_start = gyro_data.shape[0] - window_size - 1
        prev_chord_angle = 0.0
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
                prev_chord_angle = curr_chord_angle
            else:
                prev_a = a - stride
                if prev_a < 0:
                    delta_head = 0.0
                    prev_chord_angle = curr_chord_angle
                else:
                    prev_p = pos2d[prev_a]
                    prev_diff = pa - prev_p
                    prev_chord_angle = np.arctan2(prev_diff[1], prev_diff[0])
                    delta_head = wrap_angle(curr_chord_angle - prev_chord_angle)

            y_len.append(np.array([delta_len], dtype=np.float32))
            y_head.append(np.array([delta_head], dtype=np.float32))
            if return_delta_p:
                pa3 = pos_data[a, :]
                pb3 = pos_data[b, :]
                y_dp.append((pb3 - pa3).astype(np.float32))
            if return_abs_heading:
                if abs_heading_from_ori:
                    curr_abs = float(wrap_angle(yaw_from_quat(ori_data[b])[0] - yaw0))
                else:
                    curr_abs = curr_chord_angle
                    if align_heading_to_init_pose:
                        curr_abs = float(wrap_angle(curr_abs - yaw0))
                y_abs.append(np.array([curr_abs], dtype=np.float32))
            if return_ori:
                y_ori.append(ori_data[b].astype(np.float32))
            if return_rel_ori:
                qa = ori_data[a].astype(np.float32)
                qb = ori_data[b].astype(np.float32)
                q_rel = quat_mul(quat_conj(qa), qb)
                y_rel.append(q_rel)

        x_gyro = np.array(x_gyro)
        x_acc = np.array(x_acc)
        y_len = np.array(y_len)
        y_head = np.array(y_head)
        if return_abs_heading:
            y_abs = np.array(y_abs)
        if return_ori:
            y_ori = np.array(y_ori)
        if return_rel_ori:
            y_rel = np.array(y_rel)
        if return_delta_p:
            y_dp = np.array(y_dp)


        if smooth_length and len(y_len) > 0:
            y_len_smooth = gaussian_filter1d(y_len.flatten(), sigma=length_sigma)
            y_len = y_len_smooth.reshape(-1, 1)

        if smooth_heading and len(y_head) > 0:
            y_head_smooth = gaussian_filter1d(y_head.flatten(), sigma=heading_sigma)
            y_head = y_head_smooth.reshape(-1, 1)
            if return_abs_heading and len(y_abs) > 0:
                sin_h = np.sin(y_abs.flatten())
                cos_h = np.cos(y_abs.flatten())
                sin_s = gaussian_filter1d(sin_h, sigma=heading_sigma)
                cos_s = gaussian_filter1d(cos_h, sigma=heading_sigma)
                y_abs = np.arctan2(sin_s, cos_s).reshape(-1, 1)

        if return_abs_heading and return_ori and return_rel_ori and return_delta_p:
            return [x_gyro, x_acc], [y_len, y_head, y_abs, y_ori, y_rel, y_dp], init_pos, init_head
        if return_abs_heading and return_rel_ori and return_delta_p:
            return [x_gyro, x_acc], [y_len, y_head, y_abs, y_rel, y_dp], init_pos, init_head
        if return_abs_heading and return_delta_p:
            return [x_gyro, x_acc], [y_len, y_head, y_abs, y_dp], init_pos, init_head
        if return_rel_ori and return_delta_p:
            return [x_gyro, x_acc], [y_len, y_head, y_rel, y_dp], init_pos, init_head
        if return_delta_p:
            return [x_gyro, x_acc], [y_len, y_head, y_dp], init_pos, init_head
        if return_abs_heading and return_ori and return_rel_ori:
            return [x_gyro, x_acc], [y_len, y_head, y_abs, y_ori, y_rel], init_pos, init_head
        if return_abs_heading and return_rel_ori:
            return [x_gyro, x_acc], [y_len, y_head, y_abs, y_rel], init_pos, init_head
        if return_abs_heading:
            return [x_gyro, x_acc], [y_len, y_head, y_abs], init_pos, init_head
        if return_ori:
            return [x_gyro, x_acc], [y_len, y_head, y_ori], init_pos, init_head
        if return_rel_ori:
            return [x_gyro, x_acc], [y_len, y_head, y_rel], init_pos, init_head
        return [x_gyro, x_acc], [y_len, y_head], init_pos, init_head

    raise ValueError("mode must be '2d' or '3d'")
