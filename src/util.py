import numpy as np

def generate_trajectory(init_p, init_h, delta_l_list, delta_h_list):
    """
    使用步长（Δl）与航向角增量（Δψ）在平面内重建轨迹
    """
    trajectory = [init_p.copy()]
    current_p = init_p.copy()
    current_h = init_h

    delta_l_list = np.squeeze(delta_l_list)      # (N, 1) → (N,)
    delta_h_list = np.squeeze(delta_h_list)  # (N, 1) → (N,)

    for dl, dh in zip(delta_l_list, delta_h_list):
        # 若仍是 array([x])，则 item() 提取纯标量
        if hasattr(dl, 'item'):
            dl = dl.item()
        if hasattr(dh, 'item'):
            dh = dh.item()

        current_h = (current_h + dh + np.pi) % (2 * np.pi) - np.pi
        dx = dl * np.cos(current_h)
        dy = dl * np.sin(current_h)

        current_p = current_p + np.array([dx, dy])
        trajectory.append(current_p.copy())

    return np.array(trajectory)

def generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q):
    """
    使用四元数增量在6D空间中重建轨迹
    """
    cur_p = np.array(init_p)
    cur_q = np.array(init_q)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for y_delta_p, y_delta_q in zip(y_delta_p, y_delta_q):
        cur_p = cur_p + np.matmul(cur_q.rotation_matrix, y_delta_p.T).T
        cur_q = cur_q * Quaternion(y_delta_q)
        pred_p.append(np.array(cur_p))

    return np.reshape(pred_p, (len(pred_p), 3))
    
