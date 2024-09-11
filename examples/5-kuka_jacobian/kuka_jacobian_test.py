import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np

from utils import *
from kuka_jacobian import compute_next_joint_angles

arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]

if __name__ == '__main__':
    pb_client = initializeGUI(enable_gui=False)

    ## Load urdf models
    plane_id = pb_client.loadURDF("plane.urdf")
    robot_id = pb.loadURDF('kuka_iiwa/model.urdf', [0, 0, 0], useFixedBase=True)

    draw_frame(pb_client, robot_id, arm_joint_indices[-1])

    position, orientation = get_end_effector_pose(pb_client,
                                                  robot_id,
                                                  end_effector_link_index=arm_joint_indices[-1])
    debug_text_id = pb_client.addUserDebugText('{:.2f}, {:.2f}, {:.2f}'.format(*position),
                                               textPosition=(0,0,.2),
                                               textColorRGB=(0,0,0),
                                               parentObjectUniqueId=robot_id,
                                               parentLinkIndex=arm_joint_indices[-1])

    ## Main loop
    for _ in range(100000):
        
        ## Inverse kinematics (numerical solution)
        linear_jacobian, angular_jacobian = get_jacobian(pb_client, robot_id, arm_joint_indices[-1], arm_joint_indices)
        joint_angles = get_joint_angles(pb_client, robot_id, arm_joint_indices)
        position, orientation = get_end_effector_pose(pb_client,
                                                      robot_id,
                                                      end_effector_link_index=arm_joint_indices[-1])

        J_f = np.matrix(linear_jacobian)
        joint_angles = compute_next_joint_angles(joint_angles, position, J_f)

        ## Joint angle control
        for i,joint_index in enumerate(arm_joint_indices):
            joint_angle_control(pb_client, robot_id, joint_index, joint_angles[i])

        position, orientation = get_end_effector_pose(pb_client, robot_id, arm_joint_indices[-1])
        debug_text_id = pb_client.addUserDebugText('{:.2f}, {:.2f}, {:.2f}'.format(*position),
                                                   textPosition=(0,0,.2),
                                                   textColorRGB=(0,0,0),
                                                   parentObjectUniqueId=robot_id,
                                                   parentLinkIndex=arm_joint_indices[-1],
                                                   replaceItemUniqueId=debug_text_id)

        pb_client.stepSimulation()
        time.sleep(1/240)

    pb.disconnect()
