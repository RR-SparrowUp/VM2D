#loss.py
import tensorflow as tf
import numpy as np
from utils import estimate_intrinsics, project_3d_to_2d

def translation_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def rotation_loss(y_true, y_pred):
    if y_true.shape.rank is not None and y_true.shape[-1] == 9:
        y_true = tf.reshape(y_true, (-1, 3, 3))
    frob_norm = tf.norm(y_true - y_pred, ord='fro', axis=[-2, -1])
    return tf.reduce_mean(frob_norm)

def compute_reprojection_loss_numpy(kps_2d_gt, kps_3d_gt, R_gt, t_gt,
                                    kps_2d_pred, kps_3d_pred, R_pred, t_pred):
    N = int(kps_2d_gt.size // 2)
    kps_2d_gt = kps_2d_gt.reshape((N, 2))
    kps_3d_gt = kps_3d_gt.reshape((N, 3))
    R_gt = R_gt.reshape((3, 3))
    t_gt = t_gt.flatten()
    kps_2d_pred = kps_2d_pred.reshape((N, 2))
    kps_3d_pred = kps_3d_pred.reshape((N, 3))
    R_pred = R_pred.reshape((3, 3))
    t_pred = t_pred.flatten()
    K_pred = estimate_intrinsics(kps_3d_pred, kps_2d_pred, R_pred, t_pred)
    K_gt = estimate_intrinsics(kps_3d_gt, kps_2d_gt, R_gt, t_gt)
    proj_2d_pred = project_3d_to_2d(kps_3d_pred, K_pred)
    proj_2d_gt = project_3d_to_2d(kps_3d_gt, K_gt)
    loss_val = np.linalg.norm(proj_2d_pred - proj_2d_gt)
    return np.array(loss_val, dtype=np.float32)

def reprojection_loss(y_true, y_pred):
    # For the predicted branch, the model does not produce 3D keypoints.
    # Therefore, we use the ground truth "input_3d" for both branches.
    loss_val = tf.numpy_function(
        compute_reprojection_loss_numpy,
        [tf.reshape(y_true["input_2d"][0], (-1,)),
         tf.reshape(y_true["input_3d"][0], (-1,)),
         y_true["rotation"][0],
         y_true["translation"][0],
         tf.reshape(y_pred["input_2d"][0], (-1,)),
         tf.reshape(y_true["input_3d"][0], (-1,)),  # Use ground truth 3D keypoints here
         y_pred["rotation"][0],
         y_pred["translation"][0]],
        tf.float32)
    loss_val.set_shape(())
    return loss_val

def geometric_loss(y_true, y_pred, alpha=1.0, beta=0.01, gamma=0.01, delta=1.0):
    # Compute rotation loss (here, the flattened difference is used)
    pred_rot_flat = tf.reshape(y_pred["rotation"], (-1, 9))
    loss_rot = tf.reduce_mean(tf.square(y_true["rotation"] - pred_rot_flat))
    
    # Orthogonality loss: enforce that the columns of the predicted rotation matrix are orthogonal.
    rot_pred_matrix = tf.reshape(y_pred["rotation"], (-1, 3, 3))
    col1 = rot_pred_matrix[:, :, 0]
    col2 = rot_pred_matrix[:, :, 1]
    col3 = rot_pred_matrix[:, :, 2]
    ortho_loss = (tf.reduce_mean(tf.square(tf.reduce_sum(col1 * col2, axis=1))) +
                  tf.reduce_mean(tf.square(tf.reduce_sum(col1 * col3, axis=1))) +
                  tf.reduce_mean(tf.square(tf.reduce_sum(col2 * col3, axis=1))))
    
    loss_trans = tf.reduce_mean(tf.square(y_true["translation"] - y_pred["translation"]))
    loss_repr = reprojection_loss(y_true, y_pred)
    
    total = alpha * loss_rot + beta * ortho_loss + gamma * loss_trans + delta * loss_repr
    return total
