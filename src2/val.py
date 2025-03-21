#val.py
import os
import json
import logging
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from vis import plot_2d_keypoints, plot_3d_keypoints
from utils import view_matrix_2_rot_trans, plot_figs_to_png, project_3d_to_2d


def plot_validation_samples_imgs(model, val_dataset, epoch, dataloader, logger, IMAGES_DIR, projection_matrix):
    """
    Pick one batch from the validation dataset and generate plots for three random samples.
    Saves plots under a dedicated folder for the current epoch.
    """
    try:
        # Get one batch of data from validation dataset
        for batch_data in val_dataset.take(1):
            normalized_kpts, normalized_view_matrix, original_view_matrix, kpts_2d, kpts_3d = batch_data
            
            # Make predictions using the model
            # Cast to float32 for model consistency
            normalized_kpts_f32 = tf.cast(normalized_kpts, tf.float32)
            predictions = model(normalized_kpts_f32)
            if isinstance(predictions, (list, tuple)) and len(predictions) >= 3:
                _, rotation_pred_batch, translation_pred_batch = predictions[:3]
            else:
                raise ValueError("Unexpected prediction format from model.")
            
            # Convert tensors to numpy arrays if needed
            kpts_2d_np = kpts_2d.numpy()
            kpts_3d_np = kpts_3d.numpy()
            rotation_pred_np = rotation_pred_batch.numpy()
            translation_pred_np = translation_pred_batch.numpy()
            
            # Select random samples from the batch
            batch_size = kpts_2d_np.shape[0]
            sample_indices = random.sample(range(batch_size), min(3, batch_size))
            
            # Create epoch folder for saving plots
            epoch_folder = os.path.join(IMAGES_DIR, f"epoch_{epoch:02d}")
            os.makedirs(epoch_folder, exist_ok=True)
            
            # Plot each sample
            for i, idx in enumerate(sample_indices):
                gt_2d = kpts_2d_np[idx]
                gt_3d = kpts_3d_np[idx]
                
                # Get predictions for this sample
                pred_rot = rotation_pred_np[idx].reshape(3, 3)
                pred_trans = translation_pred_np[idx]
                
                # Create view matrix for prediction
                view_matrix_pred = np.eye(4)
                view_matrix_pred[:3, :3] = pred_rot
                view_matrix_pred[:3, 3] = pred_trans
                
                # Project ground truth 3D points to 2D using predicted view matrix
                pred_2d = project_3d_to_2d(gt_3d, projection_matrix, view_matrix_pred)
                
                # Plot ground truth 2D keypoints
                fig_gt_2d = plot_2d_keypoints(gt_2d, dataloader.joint_order)
                gt_2d_path = os.path.join(epoch_folder, f"sample_{i:02d}_gt_2d")
                plot_figs_to_png([fig_gt_2d], gt_2d_path, dimensions=(512, 512))
                
                # Plot predicted 2D keypoints
                fig_pred_2d = plot_2d_keypoints(pred_2d, dataloader.joint_order)
                pred_2d_path = os.path.join(epoch_folder, f"sample_{i:02d}_pred_2d")
                plot_figs_to_png([fig_pred_2d], pred_2d_path, dimensions=(512, 512))
                
                # Plot ground truth 3D keypoints
                fig_gt_3d = plot_3d_keypoints(gt_3d, dataloader.joint_order)
                gt_3d_path = os.path.join(epoch_folder, f"sample_{i:02d}_gt_3d")
                plot_figs_to_png([fig_gt_3d], gt_3d_path, dimensions=(512, 512))
                
                # Transform 3D points to camera coordinates using predicted view matrix
                ones = np.ones((gt_3d.shape[0], 1))
                pts_hom = np.concatenate([gt_3d, ones], axis=-1)
                pred_3d_cam = np.matmul(pts_hom, view_matrix_pred)[:, :3]
                
                # Plot predicted 3D keypoints (in camera space)
                fig_pred_3d = plot_3d_keypoints(pred_3d_cam, dataloader.joint_order)
                pred_3d_path = os.path.join(epoch_folder, f"sample_{i:02d}_pred_3d")
                plot_figs_to_png([fig_pred_3d], pred_3d_path, dimensions=(512, 512))
                
                logger.info(f"Saved validation plots for sample {idx} of epoch {epoch}.")
    except Exception as e:
        logger.error(f"Error during validation plotting: {e}", exc_info=True)