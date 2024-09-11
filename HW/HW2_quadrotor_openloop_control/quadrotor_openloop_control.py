import pdb

import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np

from utils import *
from control_compute import compute_orientation_control_gain, compute_force_torque

## System parameters
mass = .5
gravity = 1
inertial = (0.0023, 0.0023, 0.004)
arm_length = .1750
drag_coefficient = .01
propeller_locations = [(.1750, 0, 0), (0, .1750, 0), (-.1750, 0, 0), (0, -.1750, 0)]

if __name__ == '__main__':
    ## Initialize the simulator
    pb_client = initializeGUI(gravity=-gravity)

    ## Add plane and robot models
    plane_id = pb.loadURDF("plane.urdf")
    robot_id = pb.loadURDF('./robot_models/quadrotor.urdf',
                           [0, 0, 1],
                           pb.getQuaternionFromEuler([0, 0, 0]))
    pb_client.resetBaseVelocity(objectUniqueId=robot_id, linearVelocity=[0, 0, np.random.random()])
    

    parameter_info = []
    parameter_info.append({'name': 'roll', 'lower_limit': -np.pi/4, 'upper_limit': np.pi/4, 'start_value':0})
    parameter_info.append({'name': 'pitch', 'lower_limit': -np.pi/4, 'upper_limit': np.pi/4, 'start_value':0})
    debug_parameter_ids = add_debug_parameters(pb_client, parameter_info)

    position, orientation = get_pose(pb_client, robot_id)
    debug_text_id = pb_client.addUserDebugText('{:.2f}, {:.2f}, {:.2f}'.format(*position),
                                               textPosition=(0,0,.2),
                                               textColorRGB=(0,0,0),
                                               parentObjectUniqueId=robot_id,
                                               parentLinkIndex=-1)

    ## Draw robot frame
    draw_frame(pb_client, robot_id, -1)

    ## Compute orientation control gain
    K = compute_orientation_control_gain()

    ## Main loop
    h = 1/240
    inertial_matrix = np.diag(inertial)
    for time_step in range(100000):
        robot_state = get_robot_state(pb_client, robot_id)
        
        ## Update debug message
        position, orientation = get_pose(pb_client, robot_id)
        debug_text_id = pb_client.addUserDebugText('{:.2f}, {:.2f}, {:.2f}'.format(*position),
                                                   textPosition=(0,0,.2),
                                                   textColorRGB=(0,0,0),
                                                   parentObjectUniqueId=robot_id,
                                                   parentLinkIndex=-1,
                                                   replaceItemUniqueId=debug_text_id)

        ## Read user inputs
        roll_input = pb_client.readUserDebugParameter(itemUniqueId=0)
        pitch_input = pb_client.readUserDebugParameter(itemUniqueId=1)

        ## Change the camera location to follow the robot
        set_camera(2, -30, -10, robot_state[0])
        
        ## Compute control input
        roll, pitch, yaw = robot_state[1]
        
        R_x = np.matrix([[1, 0, 0],
                         [0, np.cos(roll), -np.sin(roll)],
                         [0, np.sin(roll), np.cos(roll)]])
        R_y = np.matrix([[np.cos(pitch), 0, np.sin(pitch)],
                         [0, 1, 0],
                         [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.matrix([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0],
                         [0, 0, 1]])
        velocity_global_frame = np.squeeze(np.asarray(np.dot(R_z*R_y*R_x, robot_state[2])))

        ## Compute/apply force and torque
        control_input = compute_force_torque(K,
                                             gravity,
                                             roll_input,
                                             pitch_input,
                                             velocity_global_frame,
                                             robot_state[1],
                                             robot_state[3])

        force_torque_control(pb_client, robot_id, control_input)
    
        pb.stepSimulation()
        # time.sleep(1/240)
        
    pb.disconnect()
