#vis.py
import matplotlib.pyplot as plt
import numpy as np

def plot_3d_keypoints(data, joint_order):
    conns = [('Head','Neck'),('Neck','Chest'),('Chest','LeftShoulder'),('LeftShoulder','LeftArm'),
             ('LeftArm','LeftForearm'),('LeftForearm','LeftHand'),('Chest','RightShoulder'),
             ('RightShoulder','RightArm'),('RightArm','RightForearm'),('RightForearm','RightHand'),
             ('Hips','LeftThigh'),('LeftThigh','LeftLeg'),('LeftLeg','LeftFoot'),
             ('Hips','RightThigh'),('RightThigh','RightLeg'),('RightLeg','RightFoot'),
             ('RightHand','RightFinger'),('RightFinger','RightFingerEnd'),
             ('LeftHand','LeftFinger'),('LeftFinger','LeftFingerEnd'),
             ('Head','HeadEnd'),('RightFoot','RightHeel'),('RightHeel','RightToe'),
             ('RightToe','RightToeEnd'),('LeftFoot','LeftHeel'),('LeftHeel','LeftToe'),
             ('LeftToe','LeftToeEnd'),('SpineLow','Hips'),('SpineMid','SpineLow'),
             ('Chest','SpineMid')]
    pts = np.array(data).reshape(len(joint_order), 3)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='red')
    for j1, j2 in conns:
        if j1 in joint_order and j2 in joint_order:
            i1, i2 = joint_order.index(j1), joint_order.index(j2)
            ax.plot([pts[i1,0], pts[i2,0]],[pts[i1,1], pts[i2,1]],[pts[i1,2], pts[i2,2]], c='blue')
    return fig

def plot_2d_keypoints(data, joint_order):
    conns = [('Head','Neck'),('Neck','Chest'),('Chest','LeftShoulder'),('LeftShoulder','LeftArm'),
             ('LeftArm','LeftForearm'),('LeftForearm','LeftHand'),('Chest','RightShoulder'),
             ('RightShoulder','RightArm'),('RightArm','RightForearm'),('RightForearm','RightHand'),
             ('Hips','LeftThigh'),('LeftThigh','LeftLeg'),('LeftLeg','LeftFoot'),
             ('Hips','RightThigh'),('RightThigh','RightLeg'),('RightLeg','RightFoot'),
             ('RightHand','RightFinger'),('RightFinger','RightFingerEnd'),
             ('LeftHand','LeftFinger'),('LeftFinger','LeftFingerEnd'),
             ('Head','HeadEnd'),('RightFoot','RightHeel'),('RightHeel','RightToe'),
             ('RightToe','RightToeEnd'),('LeftFoot','LeftHeel'),('LeftHeel','LeftToe'),
             ('LeftToe','LeftToeEnd'),('SpineLow','Hips'),('SpineMid','SpineLow'),
             ('Chest','SpineMid')]
    pts = np.array(data).reshape(len(joint_order), 2)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.scatter(pts[:,0], pts[:,1], c='red')
    for j1, j2 in conns:
        if j1 in joint_order and j2 in joint_order:
            i1, i2 = joint_order.index(j1), joint_order.index(j2)
            ax.plot([pts[i1,0], pts[i2,0]],[pts[i1,1], pts[i2,1]], c='blue')
    return fig