#val.py
import os
import json
import logging
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from vis import plot_2d_keypoints, plot_3d_keypoints
from utils import view_matrix_2_rot_trans, plot_figs_to_png, estimate_intrinsics, project_3d_to_2d


def plot_validation_samples_imgs(model, val_dataset, epoch, dataloader,logger,IMAGES_DIR):
    """
    Pick one batch from the validation dataset and generate plots for three random samples.
    Saves plots under a dedicated folder for the current epoch.
    """
    try:
        inputs, targets = next(iter(val_dataset))
    except StopIteration:
        logger.warning("Validation dataset is empty.")
        return


    predictions = model.predict(inputs["input_2d"])
    if isinstance(predictions, (list, tuple)) and len(predictions) >= 3:
        K_pred_batch, rotation_pred_batch, translation_pred_batch = predictions[:3]
    else:
        raise ValueError("Unexpected prediction format from model.")

    # Convert tensors to numpy arrays.
    input_2d_np = inputs["input_2d"].numpy()
    input_3d_np = inputs["input_3d"].numpy()
    rotation_gt_np = targets["rotation"].numpy()
    translation_gt_np = targets["translation"].numpy()
    rotation_pred_np = rotation_pred_batch 
    translation_pred_np = translation_pred_batch  
    
    batch_size = input_2d_np.shape[0]
    sample_indices = random.sample(range(batch_size), min(3, batch_size))
    
    epoch_folder = os.path.join(IMAGES_DIR, f"epoch_{epoch:02d}")
    os.makedirs(epoch_folder, exist_ok=True)
    
    for i, idx in enumerate(sample_indices):
        gt_2d = input_2d_np[idx].reshape(-1, 2)
        gt_3d = input_3d_np[idx].reshape(-1, 3)
        
        pred_rot = rotation_pred_np[idx]
        pred_trans = translation_pred_np[idx]

        K_pred = K_pred_batch[idx] if K_pred_batch.ndim == 3 else K_pred_batch

        X_cam_pred = (pred_rot.reshape(3, 3) @ gt_3d.T).T + pred_trans.flatten()
        x_proj_pred = project_3d_to_2d(X_cam_pred, K_pred)
        
        fig_gt_2d = plot_2d_keypoints(gt_2d, dataloader.joint_order)
        gt_2d_path = os.path.join(epoch_folder, f"sample_{i:02d}_gt_2d.png")
        plot_figs_to_png([fig_gt_2d], gt_2d_path, dimensions=(512, 512))
        
        fig_pred_2d = plot_2d_keypoints(x_proj_pred, dataloader.joint_order)
        pred_2d_path = os.path.join(epoch_folder, f"sample_{i:02d}_pred_2d.png")
        plot_figs_to_png([fig_pred_2d], pred_2d_path, dimensions=(512, 512))
        
        fig_gt_3d = plot_3d_keypoints(gt_3d, dataloader.joint_order)
        gt_3d_path = os.path.join(epoch_folder, f"sample_{i:02d}_gt_3d.png")
        plot_figs_to_png([fig_gt_3d], gt_3d_path, dimensions=(512, 512))
        
        fig_pred_3d = plot_3d_keypoints(X_cam_pred, dataloader.joint_order)
        pred_3d_path = os.path.join(epoch_folder, f"sample_{i:02d}_pred_3d.png")
        plot_figs_to_png([fig_pred_3d], pred_3d_path, dimensions=(512, 512))
        logger.info(f"Saved validation plots for sample {idx} of epoch {epoch}.")