#predict.py
import argparse
import json
import numpy as np
import tensorflow as tf

from model import build_model
from configs import model_train_config

def load_model(checkpoint_dir):
    """
    Build the model and restore the latest checkpoint.
    """
    model = build_model(model_train_config)
    # Create a checkpoint that tracks the model.
    checkpoint = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        checkpoint.restore(latest_ckpt).expect_partial()
        print(f"Model restored from {latest_ckpt}")
    else:
        print(f"No checkpoint found in {checkpoint_dir}. Exiting.")
        exit(1)
    return model

def load_input_data(input_path):
    """
    Load input data from a JSON file.
    Expected format: a JSON object with key "input_2d" that contains a list of 31 keypoints,
    where each keypoint is a list [x, y].
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    if "input_2d" not in data:
        raise ValueError("JSON file must contain an 'input_2d' key.")
    keypoints = np.array(data["input_2d"], dtype=np.float32)
    if keypoints.shape != (31, 2):
        raise ValueError(f"Input keypoints must have shape (31, 2), got {keypoints.shape}")
    return keypoints

def main():
    parser = argparse.ArgumentParser(
        description="Run prediction for rotation and translation from 31 2D keypoints"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/sample_predict.json",
        help="Path to JSON file containing input data (must include 'input_2d' key with 31 keypoints)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/",
        help="Directory containing model checkpoints (default: ./checkpoints/)"
    )
    args = parser.parse_args()

    # Load and process the input 2D keypoints.
    keypoints_2d = load_input_data(args.input)
    input_data = keypoints_2d.flatten()[np.newaxis, :]


    model = load_model(args.checkpoint_dir)


    predictions = model.predict(input_data)
    if isinstance(predictions, (list, tuple)) and len(predictions) >= 3:
        K_pred, rotation_pred, translation_pred = predictions[:3]
    else:
        raise ValueError("Unexpected prediction format from model.")

    # Print the predicted rotation and translation.
    print("Predicted Rotation:")
    print(rotation_pred)
    print("Predicted Translation:")
    print(translation_pred)

if __name__ == '__main__':
    main()
