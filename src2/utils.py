#utils.py

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

def estimate_intrinsics(X_3d, x_2d, R, t):
    t = t.flatten()
    N = X_3d.shape[0]
    M = (R @ X_3d.T).T + t
    if np.median(M[:,2]) < 0:
        M = -M
    A_h, b_h, A_v, b_v = [], [], [], []
    for i in range(N):
        m1, m2, m3 = M[i]
        u, v = x_2d[i]
        A_h.append([m1, m3]); b_h.append(u*m3)
        A_v.append([m2, m3]); b_v.append(v*m3)
    A_h = np.array(A_h); b_h = np.array(b_h)
    A_v = np.array(A_v); b_v = np.array(b_v)
    sol_h, _, _, _ = lstsq(A_h, b_h)
    sol_v, _, _, _ = lstsq(A_v, b_v)
    f_x, c_x = sol_h; f_y, c_y = sol_v
    K = np.array([[abs(f_x), 0, c_x],[0, abs(f_y), c_y],[0,0,1]])
    return K

def project_3d_to_2d(points_3d, projection_matrix, view_matrix=None, viewport_width=1080, viewport_height=1980):
    # Convert TensorFlow tensors to NumPy arrays if needed
    if isinstance(points_3d, tf.Tensor):
        points_3d = points_3d.numpy()
    if isinstance(projection_matrix, tf.Tensor):
        projection_matrix = projection_matrix.numpy()
    if view_matrix is not None and isinstance(view_matrix, tf.Tensor):
        view_matrix = view_matrix.numpy()
        
    if view_matrix is not None:
        ones = np.ones_like(points_3d[..., :1])
        points_hom = np.concatenate([points_3d, ones], axis=-1)
        points_camera = np.matmul(points_hom, view_matrix)
    else:
        points_camera = points_3d
        if np.median(points_camera[:,2]) < 0:
            points_camera = -points_camera
    
    points_clip = np.matmul(points_camera, projection_matrix)
    w = points_clip[..., 3:4]
    points_ndc = points_clip[..., :3] / w
    
    viewport_x, viewport_y = 0, 0
    x_ndc = points_ndc[..., 0]
    y_ndc = points_ndc[..., 1]
    
    x_screen = (x_ndc * 0.5 + 0.5) * viewport_width + viewport_x
    y_screen = (y_ndc * 0.5 + 0.5) * viewport_height + viewport_y
    
    return np.stack([x_screen, y_screen], axis=-1)

def plot_figs_to_png(figs, filename , dimensions=(512, 512)):
    plt.figure(figsize=dimensions)
    for i, fig in enumerate(figs):
        fig.savefig(f"{filename}_{i}.png")
        plt.close(fig)

def view_matrix_2_rot_trans(view_matrix):
    if isinstance(view_matrix, tf.Tensor):
        R = view_matrix[:3, :3]
        t = view_matrix[:3, 3]
    else:
        R = view_matrix[:3, :3]
        t = view_matrix[:3, 3]
    return R, t