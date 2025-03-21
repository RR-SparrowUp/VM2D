#train.py

import os
import json
import logging
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from model import build_model
from dataloader import PoseDataLoader
from loss import reprojection_loss, geometric_loss, translation_loss, rotation_loss
from utils import view_matrix_2_rot_trans, plot_figs_to_png, project_3d_to_2d
from vis import plot_2d_keypoints, plot_3d_keypoints
from val import plot_validation_samples_imgs

# Define log, checkpoint, and images directories.
BASE_LOG_DIR = os.path.join("logs", "name_trial")
CHECKPOINT_DIR = os.path.join(BASE_LOG_DIR, "checkpoints")
IMAGES_DIR = os.path.join(BASE_LOG_DIR, "images")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Set up logging: log to both console and a file.
log_file = os.path.join(BASE_LOG_DIR, "train.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

with open('config_train.json', 'r') as f:
    config = json.load(f)

def main():
    # Create the dataloader instance.
    dataloader = PoseDataLoader(config["TRAIN_DATA_PATH"], config["VAL_DATA_PATH"])
    logger.info("Data loaded successfully.")

    # Training parameters.
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    # Override checkpoint directory to be within the logs folder.
    config["checkpoint_dir"] = CHECKPOINT_DIR

    # Prepare the training and validation datasets.
    train_dataset = dataloader.prepare_data(batch_size=batch_size, is_training=True)
    val_dataset = dataloader.prepare_data(batch_size=batch_size, is_training=False)
    logger.info("Training and validation datasets prepared.")

    # Build the model.
    model = build_model(config)
    #model.summary(print_fn=logger.info)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Set up checkpointing.
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=3)
    # Convert projection matrix to a TensorFlow tensor
    projection_matrix = tf.constant(config["projection_matrix"], dtype=tf.float32)
    projection_matrix = tf.reshape(projection_matrix, (4, 4))

    @tf.function
    def train_step(normalized_kpts, normalized_view_matrix, original_view_matrix, kpts_2d, kpts_3d):
        with tf.GradientTape() as tape:
            # Cast input to float32 for model
            normalized_kpts_f32 = tf.cast(normalized_kpts, tf.float32)
            predictions = model(normalized_kpts_f32, training=True)
            K_pred, rotation_pred, translation_pred = predictions
            
            # Extract rotation and translation from normalized view matrix
            R_gt, t_gt = view_matrix_2_rot_trans(normalized_view_matrix[0])
            
            # Create target and prediction dictionaries for loss computation
            # Cast ground truth components to float32 to match model outputs
            target_dict = {
                "input_2d": tf.cast(kpts_2d, tf.float32),
                "input_3d": tf.cast(kpts_3d, tf.float32),
                "rotation": tf.cast(tf.reshape(R_gt, (1, 9)), tf.float32),
                "translation": tf.cast(tf.reshape(t_gt, (1, 3)), tf.float32),
            }
            
            pred_dict = {
                "input_2d": tf.cast(kpts_2d, tf.float32),
                "K": projection_matrix,
                "rotation": rotation_pred,
                "translation": translation_pred,
            }
            
            rep_loss = reprojection_loss(target_dict, pred_dict)
            geo_loss = geometric_loss(target_dict, pred_dict)
            trans_loss = translation_loss(target_dict["translation"], pred_dict["translation"])
            rot_loss = rotation_loss(target_dict["rotation"], pred_dict["rotation"])
            total_loss = rep_loss + geo_loss + trans_loss + rot_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        grads_vars = [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]
        missing_vars = [v.name for g, v in zip(gradients, model.trainable_variables) if g is None]
        if missing_vars:
            logger.info(f"No gradients for variables: {missing_vars}")
        optimizer.apply_gradients(grads_vars)
        return total_loss, rep_loss, geo_loss, trans_loss, rot_loss

    # Testing reprojection functionality
    try:
        print("==Testing a single input reprojection ==")
        data = dataloader.__getitem__(0)
        kp2d = data['kp2d_camera']
        kp3d_w = data['kp3d_world']
        view_matrix = data['vm_norm_flat'].reshape(4,4)
        
        # Convert TF projection matrix to numpy for testing
        projection_matrix_np = projection_matrix.numpy()
        
        #plot of Original 2D keypoints
        fig = plot_2d_keypoints(kp2d, dataloader.joint_order)
        plot_figs_to_png([fig], os.path.join(IMAGES_DIR, "original_2d"), dimensions=(512, 512))
        
        #plot of Original 3D keypoints
        fig_3d_w = plot_3d_keypoints(kp3d_w, dataloader.joint_order)
        plot_figs_to_png([fig_3d_w], os.path.join(IMAGES_DIR, "original_3d"), dimensions=(512, 512))
        
        #Convert 3D keypoints to 2D using the fixed project_3d_to_2d function
        kp2d_reprojected = project_3d_to_2d(kp3d_w, projection_matrix_np, view_matrix)
        
        #Plot of Reprojected 2D keypoints
        fig_2d_reproj = plot_2d_keypoints(kp2d_reprojected, dataloader.joint_order)
        plot_figs_to_png([fig_2d_reproj], os.path.join(IMAGES_DIR, "reprojected_2d"), dimensions=(512, 512))
    except Exception as e:
        logger.error(f"Error during reprojection test: {e}")

    global_step = 0
    try:
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            # Don't try to calculate the dataset length - TF datasets don't support it directly
            progress_bar = tqdm(train_dataset, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch_data in enumerate(progress_bar):
                normalized_kpts, normalized_view_matrix, original_view_matrix, kpts_2d, kpts_3d = batch_data
                
                total_loss, rep_loss, geo_loss, trans_loss, rot_loss = train_step(
                    normalized_kpts, normalized_view_matrix, original_view_matrix, kpts_2d, kpts_3d
                )
                
                global_step += 1
                progress_bar.set_postfix({
                    "Total": f"{total_loss.numpy():.2f}",
                    "Reproj": f"{rep_loss.numpy():.2f}",
                    "Geo": f"{geo_loss.numpy():.2f}",
                    "Trans": f"{trans_loss.numpy():.2f}",
                    "Rot": f"{rot_loss.numpy():.2f}"
                })
                
            ckpt_save_path = checkpoint_manager.save()
            logger.info(f"Checkpoint saved at {ckpt_save_path}")

            # Run validation plotting for 3 random samples and save under a dedicated epoch folder.
            try:
                # Convert projection matrix to numpy for validation functions
                projection_matrix_np = projection_matrix.numpy()
                
                # Use the custom validation function to generate plots
                plot_validation_samples_imgs(model, val_dataset, epoch+1, dataloader, logger, IMAGES_DIR, projection_matrix_np)
                
                # Also do in-line validation for more samples
                for batch_data in val_dataset.take(1):
                    normalized_kpts, normalized_view_matrix, original_view_matrix, kpts_2d, kpts_3d = batch_data
                    
                    # Cast to float32 for model consistency
                    normalized_kpts_f32 = tf.cast(normalized_kpts, tf.float32)
                    predictions = model(normalized_kpts_f32, training=False)
                    K_pred_batch, rotation_pred_batch, translation_pred_batch = predictions
                    
                    batch_size = normalized_kpts.shape[0]
                    sample_indices = random.sample(range(batch_size), min(3, batch_size))
                    
                    epoch_folder = os.path.join(IMAGES_DIR, f"epoch_{epoch+1:02d}")
                    os.makedirs(epoch_folder, exist_ok=True)
                    
                    for i, idx in enumerate(sample_indices):
                        gt_2d = kpts_2d[idx].numpy()
                        gt_3d = kpts_3d[idx].numpy()
                        
                        pred_rot = rotation_pred_batch[idx].numpy()
                        pred_trans = translation_pred_batch[idx].numpy()
                        
                        # Create view matrix for prediction
                        view_matrix_pred = np.eye(4)
                        view_matrix_pred[:3, :3] = pred_rot.reshape(3, 3)
                        view_matrix_pred[:3, 3] = pred_trans
                        
                        # Project ground truth 3D points using predicted view matrix
                        pred_2d = project_3d_to_2d(gt_3d, projection_matrix_np, view_matrix_pred)
                        
                        # Plot ground truth 2D
                        fig_gt_2d = plot_2d_keypoints(gt_2d, dataloader.joint_order)
                        plot_figs_to_png([fig_gt_2d], os.path.join(epoch_folder, f"sample_{i:02d}_gt_2d"), dimensions=(512, 512))
                        
                        # Plot predicted 2D
                        fig_pred_2d = plot_2d_keypoints(pred_2d, dataloader.joint_order)
                        plot_figs_to_png([fig_pred_2d], os.path.join(epoch_folder, f"sample_{i:02d}_pred_2d"), dimensions=(512, 512))
                        
                        # Plot ground truth 3D
                        fig_gt_3d = plot_3d_keypoints(gt_3d, dataloader.joint_order)
                        plot_figs_to_png([fig_gt_3d], os.path.join(epoch_folder, f"sample_{i:02d}_gt_3d"), dimensions=(512, 512))
                        
                        logger.info(f"Saved validation plots for sample {idx} of epoch {epoch+1}.")
            except Exception as e:
                logger.error(f"Error during validation plotting: {e}", exc_info=True)

    except tf.errors.OutOfRangeError:
        logger.info("Reached the end of the training dataset.")
    except Exception as e:
        logger.error("An error occurred during training.", exc_info=True)


if __name__ == '__main__':
    main()
