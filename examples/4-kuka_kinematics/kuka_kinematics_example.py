import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np

from utils import *
from kinematics_exercise import *

arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]

if __name__ == '__main__':
    pb_client = initializeGUI()

    ## Load urdf models
    plane_id = pb_client.loadURDF("plane.urdf")
    robot_id = pb.loadURDF('./robot_models/model.urdf', [0, 0, 0], useFixedBase=True)

    ## Set debug parameters
    parameter_info = []

    for joint_index in arm_joint_indices:
        parameter_info.append({'name': 'Joint {}'.format(joint_index+1), 'lower_limit': -180, 'upper_limit': 180, 'start_value':0})

    debug_parameter_ids = add_debug_parameters(pb_client, parameter_info)

    draw_frame(pb_client, robot_id, arm_joint_indices[-1])

    position, orientation = get_end_effector_pose(pb_client,
                                                  robot_id,
                                                  end_effector_link_index=arm_joint_indices[-1])
    joint_angles = get_joint_angles(pb_client, robot_id, arm_joint_indices)
    estimated_position = coordinate_transform(joint_angles)
    
    debug_text_id = pb_client.addUserDebugText('measured: {:.3f}, {:.3f}, {:.3f}'.format(*position),
                                               textPosition=(0,0,1.3),
                                               textColorRGB=(0,0,0),
                                               parentObjectUniqueId=robot_id)

    debug_text_kinematics_id = pb_client.addUserDebugText('estimated: {:.3f}, {:.3f}, {:.3f}'.format(*estimated_position),
                                                          textPosition=(0,0,1.4),
                                                          textColorRGB=(0,0,0),
                                                          parentObjectUniqueId=robot_id)

    ## Main loop
    for _ in range(100000):
        for i, joint_index in enumerate(arm_joint_indices):
            ## Read joint angle data from sliding bars
            joint_value = pb_client.readUserDebugParameter(itemUniqueId=debug_parameter_ids[i])
            joint_value *= np.pi/180
            joint_angle_control(pb_client, robot_id, joint_index, joint_value)

        ## Update text display
        position, orientation = get_end_effector_pose(pb_client,
                                                      robot_id,
                                                      end_effector_link_index=arm_joint_indices[-1])
        joint_angles = get_joint_angles(pb_client, robot_id, arm_joint_indices)
        estimated_position = coordinate_transform(joint_angles)
    
        debug_text_id = pb_client.addUserDebugText('measured: {:.3f}, {:.3f}, {:.3f}'.format(*position),
                                                   textPosition=(0,0,1.3),
                                                   textColorRGB=(0,0,0),
                                                   parentObjectUniqueId=robot_id,
                                                   replaceItemUniqueId=debug_text_id)

        debug_text_kinematics_id = pb_client.addUserDebugText('computed: {:.3f}, {:.3f}, {:.3f}'.format(*estimated_position),
                                                              textPosition=(0,0,1.4),
                                                              textColorRGB=(0,0,0),
                                                              parentObjectUniqueId=robot_id,
                                                              replaceItemUniqueId=debug_text_kinematics_id)

        pb_client.stepSimulation()
        time.sleep(1/240)

    pb.disconnect()
