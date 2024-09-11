import os, time
import numpy as np

import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc

from utils import *

from manipulator import manipulator_controller, task_allocator

arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
robot_base_location = [.5, 0, 0]
tray_red_location = [-.9,.7,.3]
tray_green_location = [0,.7,.3]
tray_blue_location = [.9,.7,.3]
tray_width = 1.5*.6

if __name__ == '__main__':
    ## Initialize the simulator
    pb_client = initializeGUI(enable_gui=False)

    ## Load urdf models
    plane_id = pb_client.loadURDF("plane.urdf")
    robot_id = pb_client.loadURDF('./robot_models/kuka_vacuum.urdf',
                                  [0, 0, .5], useFixedBase=True)
    # free_joint_torques(pb_client, robot_id, arm_joint_indices)

    conveyor_id = pb_client.loadURDF('./robot_models/conveyor.urdf', robot_base_location, useFixedBase=True)

    tray_red = pb_client.loadURDF("./robot_models/traybox_red.urdf", tray_red_location, globalScaling=1.5)
    tray_green = pb_client.loadURDF("./robot_models/traybox_green.urdf", tray_green_location, globalScaling=1.5)
    tray_blue = pb_client.loadURDF("./robot_models/traybox_blue.urdf", tray_blue_location, globalScaling=1.5)

    table_id = pb_client.loadURDF(os.path.join(pybullet_data.getDataPath(),"table/table.urdf"), [0,.65,-.85], globalScaling=1.8)

    ## Draw a frame at the end-effector
    draw_frame(pb_client, robot_id, link_index=7, axis_length=.1, line_width=3)

    ## Debug texts
    position, orientation = get_end_effector_pose(pb_client,
                                                  robot_id)

    ## Main loop
    np.random.seed()

    place_object(pb_client, [-0.4, -.2, .7], 2)
    
    grasping_status = False
    
    trays = [tray_red_location, tray_green_location, tray_blue_location]
    
    # define the controller for the manipulator
    manip_control = manipulator_controller(pb_client, robot_id, arm_joint_indices)
    # define the task allocator for the application
    task_alloc = task_allocator(manip_control, trays, tray_width)
        
    for time_step in range(100000):
        ## Joint control
        cube_locations = get_cube_locations(pb_client)
        end_effector_pose = get_end_effector_pose(pb_client, robot_id)
        joint_angles = get_joint_angles(pb_client, robot_id)
        joint_velocities = get_joint_velocities(pb_client, robot_id)

        joint_angles_desired = np.zeros(7)

        # ## Example
        # end_effector_target_position = [0,-.5,0.75]
        # end_effector_target_orientation = [np.pi,0,0]
        # joint_angles_desired = inverse_kinematics(pb_client, robot_id, tray_blue_location, end_effector_target_orientation)
        
        joint_actuator_torques, joint_angles_desired = task_alloc.allocate(cube_locations, joint_angles, joint_velocities, end_effector_pose)


        # Control joint angles
        # for i, joint_index in enumerate(arm_joint_indices):
        #     joint_angle_control(pb_client, robot_id, joint_index, joint_angles_desired[i])
        
        # print(f'q-ref: {np.zeros_like(joint_angles_desired).shape}')
        
        # joint_actuator_torques = man_control.compute_torque_control(
        #     joint_angles, joint_velocities, np.array(joint_angles_desired), np.zeros_like(joint_angles_desired)
        # )
        # # print(joint_actuator_torques)
        # # ## Apply the joint torques
        # for i, joint_index in enumerate(arm_joint_indices):
        #     joint_torque_control(pb_client, robot_id, joint_index, joint_actuator_torques[i])

        pb_client.stepSimulation()
        time.sleep(1/240)

    pb.disconnect()
