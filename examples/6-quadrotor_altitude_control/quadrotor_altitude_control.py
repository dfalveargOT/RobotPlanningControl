import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np

from utils import *

mass = .5
arm_length = .1750
drag_coefficient = .01
propeller_locations = [(.1750, 0, 0), (0, .1750, 0), (-.1750, 0, 0), (0, -.1750, 0)]

if __name__ == '__main__':
    ## Initialize pybullet
    pb_client = initializeGUI(enable_gui=False)

    ## Load models
    plane_id = pb.loadURDF("plane.urdf")
    robot_id = pb.loadURDF('./robot_models/quadrotor.urdf', [0, 0, 1])
    draw_frame(pb_client, robot_id, -1)

    position, orientation = get_pose(pb_client, robot_id)
    debug_text_xyz_id = pb_client.addUserDebugText('x:{:.2f}, y:{:.2f}, z:{:.2f}'.format(*position),
                                               textPosition=(.1,0,.4),
                                               textColorRGB=(0,0,0),
                                               parentObjectUniqueId=robot_id,
                                               parentLinkIndex=-1)
    debug_text_rpy_id = pb_client.addUserDebugText('r:{:.2f}, p:{:.2f}, y:{:.2f}'.format(*orientation),
                                               textPosition=(.1,0,.2),
                                               textColorRGB=(0,0,0),
                                               parentObjectUniqueId=robot_id,
                                               parentLinkIndex=-1)

    ## Destination
    desired_altitude = 2
    mark_waypoint(pb_client, (0,0,desired_altitude), axis_length=.1, line_width=2)

    ## Main loop
    for time_step in range(100000):
        ## Get robot state
        position, orientation = get_pose(pb_client, robot_id)
        linear_velocity_local_frame, angular_velocity_local_frame = get_velocity(pb_client, robot_id)
        
        debug_text_xyz_id = pb_client.addUserDebugText('x:{:.2f}, y:{:.2f}, z:{:.2f}'.format(*position),
                                                       textPosition=(.1,0,.4),
                                                       textColorRGB=(0,0,0),
                                                       parentObjectUniqueId=robot_id,
                                                       parentLinkIndex=-1,
                                                       replaceItemUniqueId=debug_text_xyz_id)
        
        debug_text_rpy_id = pb_client.addUserDebugText('r:{:.2f}, p:{:.2f}, y:{:.2f}'.format(*orientation),
                                                       textPosition=(.1,0,.2),
                                                       textColorRGB=(0,0,0),
                                                       parentObjectUniqueId=robot_id,
                                                       parentLinkIndex=-1,
                                                       replaceItemUniqueId=debug_text_rpy_id)

        ##### EXERCISE #####
        ## Compute the force f_t using the altitude (position[2]), vertical velocity (linear_velocity_local_frame[2]), mass (mass), and gravity (=10) to stabilize the quadrotor at the desired altitude (desired altitude)
        
        ## Compute control force
        f_t = 0
        
        ##### /EXERCISE #####

        ## Apply force
        propeller_force = np.array([f_t/4]*4)
        propeller_control(pb_client, robot_id, propeller_force, arm_length=arm_length, drag_coefficient=drag_coefficient)
    
        pb.stepSimulation()
        time.sleep(1/240)
    
    # cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    # print(cubePos,cubeOrn)
    pb.disconnect()
