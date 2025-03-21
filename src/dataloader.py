#dataloader.py
import numpy as np 
import tensorflow as tf 

class PoseDataLoader():
    def __init__(self, train_data, val_data):
        self.joint_order = ['Chest','Head','HeadEnd','Hips','LeftArm','LeftFinger',
                            'LeftFingerEnd','LeftFoot','LeftForearm','LeftHand','LeftHeel',
                            'LeftLeg','LeftShoulder','LeftThigh','LeftToe','LeftToeEnd',
                            'Neck','RightArm','RightFinger','RightFingerEnd','RightFoot',
                            'RightForearm','RightHand','RightHeel','RightLeg','RightShoulder',
                            'RightThigh','RightToe','RightToeEnd','SpineLow','SpineMid']
        self.train_data = train_data
        self.val_data = val_data
    
    def __len__(self):
        train_len = len(self.train_data) if self.train_data is not None else 0
        val_len = len(self.val_data) if self.val_data is not None else 0
        return train_len + val_len 
    
    def __getitem__(self, idx):
        data = self.train_data if self.train_data is not None else self.val_data
        sample = data[idx]
        pts_2d = self.flatten_keypoints(sample['kps_2d'])
        pts_3d = self.flatten_keypoints(sample['kps_3d'])
        cam_params = self.process_camera_matrix(sample['camera_matrix'])
        rotation_target = cam_params[:9]
        translation_target = cam_params[9:]
        return {"input_2d": pts_2d, "input_3d": pts_3d}, {"rotation": rotation_target, "translation": translation_target}, {"joint_order": self.joint_order}
    def flatten_keypoints(self, keypoints_dict):
        arr = []
        for joint in self.joint_order:
            if joint in keypoints_dict and joint not in ['date','Body']:
                arr.extend(keypoints_dict[joint])
        return np.array(arr, dtype=np.float32)
    def normalize_keypoints(self, keypoints):
        return keypoints
    def process_camera_matrix(self, cam_mat):
        cam_mat = np.array(cam_mat, dtype=np.float32).reshape(4,4)
        rotation = cam_mat[:3, :3].flatten()
        translation = cam_mat[3, :3].flatten() if np.allclose(cam_mat[:3,3], 0, atol=1e-6) else cam_mat[:3,3].flatten()
        return np.concatenate([rotation, translation])
    def prepare_data(self, batch_size=32, use_train=True):
        data = self.train_data if (use_train or self.val_data is None) else self.val_data
        pts_2d = np.array([self.flatten_keypoints(item['kps_2d']) for item in data])
        pts_3d = np.array([self.flatten_keypoints(item['kps_3d']) for item in data])
        cam_params = np.array([self.process_camera_matrix(item['camera_matrix']) for item in data], dtype=np.float32)
        pts_2d = self.normalize_keypoints(pts_2d)
        rotation_targets = cam_params[:, :9]
        translation_targets = cam_params[:, 9:]
        ds = tf.data.Dataset.from_tensor_slices(({"input_2d": pts_2d, "input_3d": pts_3d},
                                                   {"rotation": rotation_targets, "translation": translation_targets}))
        if self.train_data is not None:
            ds = ds.shuffle(buffer_size=1000)
        return ds.batch(batch_size)