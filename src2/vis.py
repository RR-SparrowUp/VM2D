#Visualization script for VMP2D
#Author: Reiyo 
#Organization: Sparrow Golf
#Date: 2025-03-21
#Description: This is a visualization script for VMP2D.
#This is the script to visualize the 3D keypoints and the 2D keypoints provided a data object along with the joint order.

import matplotlib.pyplot as plt
import numpy as np
import logging
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Configure logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def plot_3d_keypoints(data, joint_order):
    """
    Plot the 3D keypoints.
    Args:
        data: The data to plot.
        joint_order: The joint order.
    Returns:
        The plot."""
    
    #connections between the joints
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
    """
    Plot the 2D keypoints.
    Args:
        data: The data to plot.
        joint_order: The joint order.
    Returns:
        The plot.
    """
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
