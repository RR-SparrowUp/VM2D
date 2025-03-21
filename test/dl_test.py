#Test Script for the dataloader.py 
#Author: Reiyo 
#Organization: Sparrow Golf
#Date: 2025-03-21
#Description: This is a test scritpt for the dataloader of VMP2D.
#It loads the data from the json file and returns a dataset.
#Make sure to check the imports and the current directory path. It should be pointing to VM2D for this script to run.

import json 
import numpy as np 
import os 
import sys 
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src2.dataloader import PoseDataLoader
from src2.vis import plot_2d_keypoints , plot_3d_keypoints

with open('config.json', 'r') as f:
    config = json.load(f)

train_file = config['train_data']
val_file = config['val_data']

dataloader = PoseDataLoader(train_file, val_file)
print(dataloader.__getitem__(0))

print("\n==Testing a single batch==")
train_data = dataloader.prepare_data(batch_size=1, is_training=True)
for i in next(iter(train_data)):
    batch = i
    print(batch)

print("\n============================\n")

print("\n==Testing a single input reprojection ==\n")
data = dataloader.__getitem__(0)
kp2d = data['kp2d_camera']
kp3d_w = data['kp3d_world']
view_matrix = data['vm_norm_flat'].reshape(4,4)
projection_matrix = np.array([ 2.790552 ,  0.,         0.,         0.,         0.,         1.5696855,
  0.,         0.,         0.,         0.,        -1.0001999, -1.,
  0.,         0.,        -0.20002,    0.,       ], dtype=np.float32)

projection_matrix = projection_matrix.reshape((4,4))
fig = plot_2d_keypoints(kp2d,dataloader.joint_order)
fig_3d_w = plot_3d_keypoints(kp3d_w,dataloader.joint_order)


ones = tf.ones_like(kp3d_w[..., :1])
points_hom = tf.concat([kp3d_w, ones], axis=-1)
points_camera = np.matmul(points_hom,view_matrix)
points_clip = tf.matmul(points_camera, projection_matrix)
w = points_clip[..., 3:4]
points_ndc = points_clip[..., :3] / w
viewport_x, viewport_y, viewport_width, viewport_height = [0,0,1080,1980]
x_ndc = points_ndc[..., 0]
y_ndc = points_ndc[..., 1]
x_screen = (x_ndc * 0.5 + 0.5) * viewport_width + viewport_x
y_screen = (y_ndc * 0.5 + 0.5) * viewport_height + viewport_y

fig_3d_c = plot_3d_keypoints(points_ndc,dataloader.joint_order)
kp2d_reprojected = tf.stack([x_screen, y_screen], axis=-1)

fig_2d_reproj = plot_2d_keypoints(kp2d_reprojected,dataloader.joint_order)

print("\n============================\n")

# #dataloader.py
# import numpy as np
# import tensorflow as tf
# import json 
# import sys 
# import os 

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, ".."))
# sys.path.insert(0, project_root)
# from src2.dataloader import PoseDataLoader

# with open('config.json', 'r') as f:
#     config = json.load(f)

# train_file = config['train_data']
# val_file = config['val_data']

# def compute_reprojection_loss(camera_matrix, points_3d, points_2d, projection, viewport):
#     """
#     Computes the reprojection loss between predicted 2D points (from 3D points)
#     and the ground truth 2D points.

#     Args:
#     camera_matrix: Predicted camera matrices, shape (batch_size, 4, 4).
#     points_3d: 3D points, shape (batch_size, num_points, 3).
#     points_2d: Ground truth 2D points, shape (batch_size, num_points, 2).

#     Returns:
#     A scalar tensor representing the reprojection loss.
#     """
#     # Convert 3D points to homogeneous coordinates: [x, y, z, 1]
#     ones = tf.ones_like(points_3d[..., :1])
#     points_hom = tf.concat([points_3d, ones], axis=-1) # shape: (batch_size, num_points, 4)

#     # Transform points by the predicted camera matrix.
#     # Use the transpose since we assume points are row vectors.
#     points_camera = tf.matmul(points_hom, camera_matrix)

#     # Apply the constant projection matrix.
#     points_clip = tf.matmul(points_camera, projection)

#     # Perform perspective division (homogeneous normalization).
#     w = points_clip[..., 3:4]
#     points_ndc = points_clip[..., :3] / w # normalized device coordinates

#     # Map normalized coordinates to viewport (screen) coordinates.
#     viewport_x, viewport_y, viewport_width, viewport_height = viewport
#     x_ndc = points_ndc[..., 0]
#     y_ndc = points_ndc[..., 1]
#     x_screen = (x_ndc * 0.5 + 0.5) * viewport_width + viewport_x
#     y_screen = (y_ndc * 0.5 + 0.5) * viewport_height + viewport_y

#     # Stack the computed 2D coordinates.
#     projected_points = tf.stack([x_screen, y_screen], axis=-1)

#     # Compute mean squared error between projected and ground truth 2D points.
#     loss = tf.sqrt(tf.reduce_sum(tf.square(projected_points - points_2d)))
#     return loss

# dataloader = PoseDataLoader(train_file, val_file)
# i = 1
# data = dataloader.__getitem__(i)
# kp2d = data['kp2d_camera']
# kp3d_w = data['kp3d_world']
# camera_matrix = data['vm']


# projection_matrix = np.array([ 2.790552 , 0., 0., 0., 0., 1.5696855,
# 0., 0., 0., 0., -1.0001999, -1.,
# 0., 0., -0.20002, 0., ], dtype=np.float32)

# projection_matrix = projection_matrix.reshape((4,4))

# ones = tf.ones_like(kp3d_w[..., :1])
# points_hom = tf.concat([kp3d_w, ones], axis=-1)
# points_camera = np.matmul(points_hom,camera_matrix)
# points_clip = tf.matmul(points_camera, projection_matrix)
# w = points_clip[..., 3:4]
# points_ndc = points_clip[..., :3] / w
# viewport_x, viewport_y, viewport_width, viewport_height = [0,0,1080,1980]
# x_ndc = points_ndc[..., 0]
# y_ndc = points_ndc[..., 1]
# x_screen = (x_ndc * 0.5 + 0.5) * viewport_width + viewport_x
# y_screen = (y_ndc * 0.5 + 0.5) * viewport_height + viewport_y

# # Stack the computed 2D coordinates.
# projected_points = tf.stack([x_screen, y_screen], axis=-1)
# print(projected_points)
# fig = plot_2d_keypoints(projected_points,dataloader.joint_order)
# print(kp2d)
# fig2 = plot_2d_keypoints(kp2d,dataloader.joint_order)

# error = np.linalg.norm(kp2d - projected_points)
# print(error)





