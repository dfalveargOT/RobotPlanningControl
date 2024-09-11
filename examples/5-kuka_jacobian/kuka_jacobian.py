import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np

def compute_next_joint_angles(joint_angles, end_effector_position, J_f):
    end_effector_destination = (.5, -.5, .5)

    ##### EXERCISE #####
    ## Given the joint angles, end effector position, and Jacobian matrix, compute the next manipulator joint angles (you will need to compute the peudo-inverse of the Jacobina matrix)
    gain = 10
    J_f_plus = J_f.T*np.linalg.inv(J_f*J_f.T)
    # print(end_effector_position)
    joint_ee_diff = J_f_plus*np.matrix(end_effector_destination - end_effector_position).T
    next_joint_angles = joint_angles + gain*joint_ee_diff.T
    next_joint_angles = np.squeeze(np.asarray(next_joint_angles))
    # next_joint_angles = [0,0,0,0,0,0,0]
    ##### /EXERCISE #####
        
    return next_joint_angles
