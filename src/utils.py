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

def project_3d_to_2d(X_cam, K, epsilon=1e-6):
    if np.median(X_cam[:,2]) < 0:
        X_cam = -X_cam
    x_hom = (K @ X_cam.T).T
    return x_hom[:, :2] / (x_hom[:, 2:3] + epsilon)

def plot_figs_to_png(figs, filename , dimensions=(512, 512)):
    plt.figure(figsize=dimensions)
    for i, fig in enumerate(figs):
        fig.savefig(f"{filename}_{i}.png")
        plt.close(fig)

def view_matrix_2_rot_trans(view_matrix):
    R = view_matrix[:3, :3]
    t = view_matrix[:3, 3]
    return R, t