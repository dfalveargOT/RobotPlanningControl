import numpy as np
from kuka_dynamics import iterative_ne_algorithm

arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]

def compute_joint_torque(time_sec, joint_angles, joint_velocities):

    h = 1/240
    ########## EXERCISE ##########
    ## Dynamic model, tau = M(theta) theta_ddot + V(theta, theta_dot) + G(theta), is implemented as
    ## tau = iterative_ne_algorithm(len(arm_joint_indices), theta, theta_dot, theta_ddot)
    
    pi_2 = np.pi/2
    theta_dot_ref = np.array([pi_2*np.cos(time_sec+h), -pi_2*np.cos(time_sec+h), 0,0,0,0,0]) 
    theta_ddot = (theta_dot_ref - joint_velocities)/h
    joint_torques = iterative_ne_algorithm(len(arm_joint_indices), joint_angles, joint_velocities, theta_ddot)

    ########## /EXERCISE ##########
    
    return joint_torques
