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
from configs import TRAIN_DATA_PATH, VAL_DATA_PATH, model_train_config
from loss import reprojection_loss, geometric_loss, translation_loss, rotation_loss
from utils import view_matrix_2_rot_trans, plot_figs_to_png, estimate_intrinsics, project_3d_to_2d
from vis import plot_2d_keypoints, plot_3d_keypoints

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


def load_data(train_path, val_path):
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    return train_data, val_data


from val import plot_validation_samples_imgs


def main():
    # Load training and validation data.
    train_data, val_data = load_data(TRAIN_DATA_PATH, VAL_DATA_PATH)
    logger.info("Data loaded successfully.")

    # Create the dataloader instance.
    dataloader = PoseDataLoader(train_data, val_data)

    # Training parameters.
    batch_size = model_train_config["batch_size"]
    epochs = model_train_config["epochs"]
    learning_rate = model_train_config["learning_rate"]
    # Override checkpoint directory to be within the logs folder.
    model_train_config["checkpoint_dir"] = CHECKPOINT_DIR

    # Prepare the training and validation datasets.
    train_dataset = dataloader.prepare_data(batch_size=batch_size, use_train=True)
    val_dataset = dataloader.prepare_data(batch_size=batch_size, use_train=False)
    logger.info("Training and validation datasets prepared.")

    # Build the model.
    model = build_model(model_train_config)
    model.summary(print_fn=logger.info)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Set up checkpointing.
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=3)

    @tf.function
    def train_step(inputs, targets):
        target_dict = {
            "input_2d": inputs["input_2d"],
            "input_3d": inputs["input_3d"],
            "rotation": targets["rotation"],
            "translation": targets["translation"],
        }
        with tf.GradientTape() as tape:
            predictions = model(inputs["input_2d"], training=True)
            K_pred, rotation_pred, translation_pred = predictions
            pred_dict = {
                "input_2d": inputs["input_2d"],
                "K": K_pred,
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

    global_step = 0
    try:
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            total_steps = len(train_dataset) if hasattr(train_dataset, "__len__") else None
            progress_bar = tqdm(train_dataset, total=total_steps, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                total_loss, rep_loss, geo_loss, trans_loss, rot_loss = train_step(inputs, targets)
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
            plot_validation_samples_imgs(model, val_dataset, epoch+1, dataloader,logger,IMAGES_DIR)

    except tf.errors.OutOfRangeError:
        logger.info("Reached the end of the training dataset.")
    except Exception as e:
        logger.error("An error occurred during training.", exc_info=True)


if __name__ == '__main__':
    main()
