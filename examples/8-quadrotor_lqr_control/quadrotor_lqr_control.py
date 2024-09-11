import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np

from utils import *
from lqr_control_exercise import *

## System parameters
mass = .5
inertial = (0.0023, 0.0023, 0.004)
arm_length = .1750
drag_coefficient = .01
propeller_locations = [(.1750, 0, 0), (0, .1750, 0), (-.1750, 0, 0), (0, -.1750, 0)]

if __name__ == '__main__':
    ## Initialize the simulator
    pb_client = initializeGUI(enable_gui=False)

    ## Add plane and robot models
    plane_id = pb.loadURDF("plane.urdf")
    robot_id = pb.loadURDF('./robot_models/quadrotor.urdf', [1, 1, 0.1])

    ## Draw robot frame
    draw_frame(pb_client, robot_id, -1)

    ## Initialize debug texts
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

    ## Set desired altitude and destination
    destination = [0, 0, 1]
    mark_waypoint(pb_client, destination, axis_length=.2, line_width=3)

    ## Compute the control gain
    control_gain = compute_control_gain()
    
    ## Main loop
    for time_step in range(100):
        pb.stepSimulation()
        time.sleep(1/240)
        
    for time_step in range(100000):

        ## Update debug texts
        position, orientation = get_pose(pb_client, robot_id)
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


        ## Compute control input (state feedback)
        robot_state = get_robot_state(pb_client, robot_id)
        
        control_input = compute_stabilizing_feedback(control_gain, robot_state, destination)
        force_torque_control(pb_client, robot_id, control_input)
    
        pb.stepSimulation()
        # time.sleep(1/240)
    
    pb.disconnect()
