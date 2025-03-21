#model.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Add, Lambda, LayerNormalization, Concatenate, Flatten, Reshape, MultiHeadAttention
from rot import sixd_to_rotmat

def build_model(config):

    inputs = Input(shape=(config['input_dim'],), name='Input_Layer')

    # === Common dense branch ===
    x = Dense(config['n_units'], activation='tanh', name='dense1')(inputs)
    x = LayerNormalization(name='ln1')(x)
    x = Dense(config['n_units'], activation='tanh', name='dense2')(x)
    x = LayerNormalization(name='ln2')(x)
    x = Dropout(config['dropout'], name='dropout1')(x)
    
    # Residual block
    x_shortcut = x
    x = Dense(config['n_units'], activation='relu', name='dense3')(x)
    x = LayerNormalization(name='ln3')(x)
    x = Dense(config['n_units'], activation='relu', name='dense4')(x)
    x = LayerNormalization(name='ln4')(x)
    x = Dropout(config['dropout_rate'], name='dropout2')(x)
    if config['residual']:
        x = Add(name='residual_add')([x_shortcut, x])
    
    common_feature_branch = x

    # === Multi-head Attention branch with positional embedding ===

    num_joints = config['num_joints']
    x_reshaped = Reshape((num_joints, 2), name='reshape_2d_keypoints')(inputs)
    joint_embedding = Dense(config['embed_dim'], activation='relu', name='joint_embedding')(x_reshaped)
    pos_embedding = Dense(config['embed_dim'], activation='relu', name='pos_embedding')(x_reshaped)
    pos_encoded_2D = Add(name='positional_encoding')([joint_embedding, pos_embedding])
    
    attn_out = MultiHeadAttention(num_heads=config['num_heads'], key_dim=config['embed_dim'], name='multi_head_attention')(
        pos_encoded_2D, pos_encoded_2D)
    
    attn_flat = Flatten(name='flatten_attention')(attn_out)
    
    combined_features = Concatenate(name='combined_features')([common_feature_branch, attn_flat])
    
    # === Branch 1: Estimate intrinsic matrix K ===
    K_out = Dense(9, activation='linear', name='K_dense')(combined_features)
    K_out = Lambda(lambda z: tf.reshape(z, (-1, 3, 3)), name='K_reshape')(K_out)
    
    # === Branch 2: Learn 3x3 rotation matrix ===
    rot_raw = Dense(6, activation='linear', name='rot_dense')(combined_features)
    rot_out = Lambda(sixd_to_rotmat, name='rot_matrix')(rot_raw)
    
    # === Branch 3: Learn translation vector ===
    trans_out = Dense(3, activation='linear', name='translation_output')(combined_features)
    
    model = Model(inputs=inputs, outputs=[K_out, rot_out, trans_out], name='VMP2D')
    
    # Note on connections:
    # The following connections list defines the relationships between joints based on their 2D positions.
    # It can be used to inform a more advanced positional embedding or graph-based module.
    # conns = [('Head','Neck'), ('Neck','Chest'), ('Chest','LeftShoulder'), ('LeftShoulder','LeftArm'),
    #          ('LeftArm','LeftForearm'), ('LeftForearm','LeftHand'), ('Chest','RightShoulder'),
    #          ('RightShoulder','RightArm'), ('RightArm','RightForearm'), ('RightForearm','RightHand'),
    #          ('Hips','LeftThigh'), ('LeftThigh','LeftLeg'), ('LeftLeg','LeftFoot'),
    #          ('Hips','RightThigh'), ('RightThigh','RightLeg'), ('RightLeg','RightFoot'),
    #          ('RightHand','RightFinger'), ('RightFinger','RightFingerEnd'),
    #          ('LeftHand','LeftFinger'), ('LeftFinger','LeftFingerEnd'),
    #          ('Head','HeadEnd'), ('RightFoot','RightHeel'), ('RightHeel','RightToe'),
    #          ('RightToe','RightToeEnd'), ('LeftFoot','LeftHeel'), ('LeftHeel','LeftToe'),
    #          ('LeftToe','LeftToeEnd'), ('SpineLow','Hips'), ('SpineMid','SpineLow'),
    #          ('Chest','SpineMid')]
    
    return model