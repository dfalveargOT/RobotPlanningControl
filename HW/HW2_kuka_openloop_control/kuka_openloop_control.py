import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np

from utils import *
from kuka_kinematics import coordinate_transform
from torque_compute import compute_joint_torque

arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]

if __name__ == '__main__':
    ## Initialize the simulator
    pb_client = initializeGUI(enable_gui=False)

    ## Load urdf models
    plane_id = pb_client.loadURDF("plane.urdf")
    robot_id = pb.loadURDF('./robot_models/model.urdf', [0, 0, 0], useFixedBase=True)
    free_joint_torques(pb_client, robot_id, arm_joint_indices)

    ## Draw a frame at the end-effector
    draw_frame(pb_client, robot_id, arm_joint_indices[-1])

    ## Debug texts
    position, orientation = get_end_effector_pose(pb_client,
                                                  robot_id,
                                                  end_effector_link_index=arm_joint_indices[-1])
    debug_text_id = pb_client.addUserDebugText('{:.2f}, {:.2f}, {:.2f}'.format(*position),
                                               textPosition=(0,0,.2),
                                               parentObjectUniqueId=robot_id,
                                               parentLinkIndex=arm_joint_indices[-1])

    ## Draw reference trajectory
    q_ref = np.array([0, 0, 0 ,0, 0, 0, 0])
    previous_end_effector_position = coordinate_transform(q_ref)
    for time_step in range(1200):
        time_sec = time_step / 240
        q_ref = np.array([np.pi/2*np.sin(time_sec), -np.pi/2*np.sin(time_sec), 0 ,0, 0, 0, 0])

        end_effector_position = coordinate_transform(q_ref)

        delta_end_effector_position = end_effector_position - previous_end_effector_position
        if(np.linalg.norm(delta_end_effector_position) >= .02):
            pb_client.addUserDebugLine(lineFromXYZ=previous_end_effector_position,
                                       lineToXYZ=end_effector_position,
                                       lineColorRGB=(.5,.5,.5),
                                       lineWidth=5)

            previous_end_effector_position = end_effector_position

    ## Main loop
    h = 1/240
    for time_step in range(100000):

        ## Update debug texts
        position, orientation = get_end_effector_pose(pb_client, robot_id, arm_joint_indices[-1])
        debug_text_id = pb_client.addUserDebugText('{:.2f}, {:.2f}, {:.2f}'.format(*position),
                                                   textPosition=(0,0,.2),
                                                   parentObjectUniqueId=robot_id,
                                                   parentLinkIndex=arm_joint_indices[-1],
                                                   replaceItemUniqueId=debug_text_id)

        ## Compute and apply the joint torques
        joint_angles, joint_velocities, _, _ = zip(*pb_client.getJointStates(bodyUniqueId=robot_id,
                                                                             jointIndices=arm_joint_indices))
        
        time_sec = time_step / 240
        joint_actuator_torques = compute_joint_torque(time_sec, joint_angles, joint_velocities)

        ## Apply the joint torques
        for i, joint_index in enumerate(arm_joint_indices):
            joint_torque_control(pb_client, robot_id, joint_index, joint_actuator_torques[i])
    
        pb_client.stepSimulation()

    pb.disconnect()
