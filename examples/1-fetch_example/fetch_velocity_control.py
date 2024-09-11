import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np

from utils import *

DEF_CONST_ANGULAR_VEL = 5
DEF_CONST_LINEAR_VEL = 1

if __name__ == '__main__':
    pb_client = initializeGUI()

    ## Load models
    plane_id = pb.loadURDF("plane.urdf")
    robot_id = pb.loadURDF('./robot_models/fetch/fetch_baseonly.urdf', [0, 0, 0])
    # robot_id = pb.loadURDF('./robot_models/fetch/fetch.urdf', [0, 0, 0])

    ## Set debug parameters
    parameter_info = []
    parameter_info.append({'name': 'Linear Velocity', 'lower_limit': -1, 'upper_limit': 1, 'start_value':0})
    parameter_info.append({'name': 'Rotational Velocity', 'lower_limit': -10, 'upper_limit': 10, 'start_value':0})
    debug_parameter_ids = add_debug_parameters(pb_client, parameter_info)
    
    set_camera(pb_client, distance=2, yaw=45, pitch=-40, position=(0,0,0))
    draw_frame(pb_client, plane_id, 0, 1, 4)
    draw_frame(pb_client, robot_id, 0, .6, 10)

    linear_velocity, angular_velocity = get_robot_velocity(pb_client, robot_id)
    debug_text_id = pb_client.addUserDebugText('{:.2f}, {:.2f}, {:.2f}'.format(linear_velocity[0], linear_velocity[1], angular_velocity[2]),
                                               textPosition=(.1,0,.5),
                                               textColorRGB=(0,0,0),
                                               parentObjectUniqueId=robot_id,
                                               parentLinkIndex=0)
    linear_velocity = DEF_CONST_LINEAR_VEL
    angular_velocity = DEF_CONST_ANGULAR_VEL

    ## Main loop
    for _ in range(100000):
        
        ## Read velocity data from sliding bars
        linear_velocity = pb_client.readUserDebugParameter(itemUniqueId=debug_parameter_ids[0])
        angular_velocity = pb_client.readUserDebugParameter(itemUniqueId=debug_parameter_ids[1])

        ## Apply control action
        apply_action(pb_client, robot_id, linear_velocity, angular_velocity)

        position, orientation = get_robot_pose(pb_client, robot_id)
        set_camera(pb_client, 3, 0, -30, position)

        ## Update text display
        linear_velocity, angular_velocity = get_robot_velocity(pb_client, robot_id)
        debug_text_id = pb_client.addUserDebugText('{:.2f}, {:.2f}, {:.2f}'.format(linear_velocity[0], linear_velocity[1], angular_velocity[2]),
                                                   textPosition=(.1,0,.5),
                                                   textColorRGB=(0,0,0),
                                                   parentObjectUniqueId=robot_id,
                                                   parentLinkIndex=0,
                                                   replaceItemUniqueId=debug_text_id)
    
        pb.stepSimulation()
        time.sleep(1/240)

    pb_client.disconnect()
