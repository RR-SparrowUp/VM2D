#configs.py

TRAIN_DATA_PATH = "/Users/ryok3n/Desktop/Workspace/VMP2D/data/train_data_20.json"
VAL_DATA_PATH = "/Users/ryok3n/Desktop/Workspace/VMP2D/data/val_data_20.json"
TEST_DATA_PATH = "/Users/ryok3n/Desktop/Workspace/VMP2D/data/test_data_20.json"

model_test_config = {
    'input_dim': 62,         # 31 joints * 2 (x,y)
    'n_units': 128,
    'dropout': 0.3,
    'dropout_rate': 0.3,
    'residual': True,
    'num_joints': 31,
    'embed_dim': 64,
    'num_heads': 4
}

model_train_config = {
    'input_dim': 62,         # 31 joints * 2 (x,y)
    'n_units': 128,
    'dropout': 0.3,
    'dropout_rate': 0.3,
    'residual': True,
    'num_joints': 31,
    'embed_dim': 64,
    'num_heads': 4,
    'batch_size': 256,
    'epochs': 10,
    'learning_rate': 0.001,
    'checkpoint_dir': "./checkpoints"
}