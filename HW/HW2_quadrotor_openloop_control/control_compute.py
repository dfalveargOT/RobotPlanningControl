import numpy as np
import control

mass = .5
inertial = (0.0023, 0.0023, 0.004)
def compute_orientation_control_gain():

    ########## EXERCISE ##########
    ## Compute LQR control gain K for orientation control
    # A = np.matrix(np.zeros((6,6)))
    # B = np.matrix(np.zeros((6,3)))

    A = np.matrix([[ 0, 0, 0, 1, 0, 0],
                [ 0, 0, 0, 0, 1, 0],
                [ 0, 0, 0, 0, 0, 1],
                [ 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0]])
    
    B = np.matrix([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [1/inertial[0], 0, 0],
                   [0, 1/inertial[1], 0],
                   [0, 0, 1/inertial[2]]])

    Q = np.identity(6)
    R = np.identity(3)
    control_gain, _, _ = control.lqr(A,B,Q,R)
    K = -control_gain

    # K = np.matrix(np.zeros((6,3)))
     
    ########## /EXERCISE ##########

    return K

def compute_force_torque(K, gravity, roll_input, pitch_input, velocity_global_frame, orientation, angular_velocity):
    h = 1/240
    roll, pitch, yaw = orientation
    roll_velocity, pitch_velocity, yaw_velocity = angular_velocity
    x_ddot, y_ddot, z_ddot = velocity_global_frame

    ########## EXERCISE ##########
    ## Compute force and torque
    x_ref = np.array([roll_input, pitch_input, 0, 0, 0, 0])
    x = np.array([roll, pitch, yaw, roll_velocity, pitch_velocity, yaw_velocity])

    c_th_c_phy = np.cos(roll)*np.cos(pitch)
    force = (1/c_th_c_phy)*((-mass*z_ddot/h) + mass*gravity)

    # print(K)
    # print((x - x_ref).T)
    torque = K.dot((x - x_ref).T)

    ########## /EXERCISE ##########

    return np.concatenate((np.array([force]), torque))
