import pybullet as pb
import pybullet_data
import time
import numpy as np

from utils import *
from exercise_velocity_control import apply_action_exercise

if __name__ == '__main__':
    pb_client = initializeGUI()

    ## Load models
    plane_id = pb.loadURDF("plane.urdf")
    robot_id = pb.loadURDF('./robot_models/cycle/tricycle.urdf', [0, 0, 1])

    ## Set debug parameters
    parameter_info = []
    parameter_info.append({'name': 'Linear Velocity', 'lower_limit': -1, 'upper_limit': 1, 'start_value':0})
    parameter_info.append({'name': 'Angular Velocity', 'lower_limit': -1, 'upper_limit': 1, 'start_value':0})
    debug_parameter_ids = add_debug_parameters(pb_client, parameter_info)

    draw_frame(pb_client, robot_id, 0, .6, 10)

    position, orientation = get_robot_pose(pb_client, robot_id)
    linear_velocity, angular_velocity = get_robot_velocity(pb_client, robot_id)
    linear_velocity_r = linear_velocity[0]*np.cos(orientation[2]) + linear_velocity[1]*np.sin(orientation[2])

    debug_text_id = pb_client.addUserDebugText('{:.2f}, {:.2f}'.format(linear_velocity_r, angular_velocity[2]),
                                                   textPosition=(.1,0,.5),
                                                   textColorRGB=(0,0,0),
                                                   parentObjectUniqueId=robot_id,
                                                   parentLinkIndex=0)
    
    
    ## Main loop
    for _ in range(100000):
        ## Read velocity data from sliding bars
        linear_velocity = pb_client.readUserDebugParameter(itemUniqueId=debug_parameter_ids[0])
        angular_velocity = pb_client.readUserDebugParameter(itemUniqueId=debug_parameter_ids[1])

        ## Apply control action (EXERCISE)
        apply_action_exercise(pb_client, robot_id, linear_velocity, angular_velocity)

        ## Update text display
        position, orientation = get_robot_pose(pb_client, robot_id)
        linear_velocity, angular_velocity = get_robot_velocity(pb_client, robot_id)
        linear_velocity_r = linear_velocity[0]*np.cos(orientation[2]) + linear_velocity[1]*np.sin(orientation[2])

        debug_text_id = pb_client.addUserDebugText('{:.2f}, {:.2f}'.format(linear_velocity_r, angular_velocity[2]),
                                                   textPosition=(.1,0,.5),
                                                   textColorRGB=(0,0,0),
                                                   parentObjectUniqueId=robot_id,
                                                   parentLinkIndex=0,
                                                   replaceItemUniqueId=debug_text_id)
        pb_client.stepSimulation()
        # time.sleep(1/240)
        
    pb_client.disconnect()
