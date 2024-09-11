import numpy as np
import control

mass = .5
inertial = (0.0023, 0.0023, 0.004) # (I_x, I_y, I_z)
gravity = 10

def compute_control_gain():
    A = np.matrix([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, gravity, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, -gravity, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    B = np.matrix([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [1/mass, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 1/inertial[0], 0, 0],
                   [0, 0, 1/inertial[1], 0],
                   [0, 0, 0, 1/inertial[2]]])
    
    ##### EXERCISE #####
    ## Compute control gain for LQR controller using control package
    control_gain = np.matrix(np.zeros((4,12)))

    ##### /EXERCISE #####

    return control_gain

def compute_stabilizing_feedback(control_gain, robot_state, destination):
    x = np.concatenate((robot_state[0],
                        robot_state[2],
                        robot_state[1],
                        robot_state[3]))

    ##### EXERCISE #####
    ## Compute control input to the quadrotor
    control_input = np.zeros(4)

    ##### /EXERCISE #####
    
    return control_input

