import argparse, time

import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc

import numpy as np

from utils import *
from racecar_mpc_se2 import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', nargs='?', const=1, type=int, default=2)
    args = parser.parse_args()
    
    ## Initialize pybullet
    pb_client = initializeGUI(False)

    ## Load models
    plane_id = pb.loadURDF("plane.urdf")
    robot_id = pb.loadURDF('./robot_models/racecar.urdf', [0, 0, 0])

    ## Draw frames
    draw_frame(pb_client, plane_id, 0, 1, 4)
    draw_frame(pb_client, robot_id, 0, .6, 10)

    ### Draw reference trajectory
    h = 1/10
    reference_trajectory = []

    if (args.trajectory == 0):
        for k in range(2000):
            t = k*h
            reference_trajectory.append((t/10, np.cos(t/10)-1, np.arctan(-np.sin(t/10))))

    elif (args.trajectory == 1):
        ## Along x-axis
        for k in range(500):
            t = k*h
            reference_trajectory.append((t/10, 0, 0))

        ## Along y-axis
        for k in range(500):
            t = k*h
            reference_trajectory.append((5, t/10, np.pi/2))

        ## Along x-axis
        for k in range(500):
            t = k*h
            reference_trajectory.append((5+t/10, 5, 0))

        ## Along y-axis
        for k in range(500):
            t = k*h
            reference_trajectory.append((10, 5-t/10, -np.pi/2))

    elif (args.trajectory == 2):
        for k in range(2000):
            t = k*h
            reference_trajectory.append((5*np.sin(t/10), 5*(1-np.cos(t/10)), t/10))
            
        
    reference_trajectory_x, reference_trajectory_y, reference_trajectory_theta = zip(*reference_trajectory)
    draw_reference_trajectory(pb_client, plane_id, reference_trajectory[::10], line_width=20)

    ## Debug text messages
    linear_velocity, angular_velocity = get_robot_velocity(pb_client, robot_id)
    # debug_text_id = pb_client.addUserDebugText('{:.2f}, {:.2f}, {:.2f}'.format(linear_velocity[0], linear_velocity[1], angular_velocity[2]),
    #                                            textPosition=(.1,0,.5),
    #                                            textColorRGB=(0,0,0),
    #                                            parentObjectUniqueId=robot_id,
    #                                            parentLinkIndex=0)

    ## Initialize MPC
    mpc_racecar = MpcRacecarRobot()
    mpc_racecar.h = h
    mpc_racecar.set_reference_trajectory(reference_trajectory)
        
    initial_time_instance = 0
    position, orientation = get_robot_pose(pb_client, robot_id)
    initial_pose = (position[0], position[1], orientation[2])
    x_initial = np.zeros(mpc_racecar.number_of_variables, float)

    ## Main loop
    for k in range(len(reference_trajectory)-100):
        ## Run MPC
        mpc_racecar.initialize_mpc(initial_time_instance, initial_pose)
        p_x, p_y, theta, v, u = mpc_racecar.compute_mpc(x_initial)
        # print(v[0], u[0])

        ## Apply control to the robot
        apply_action(pb_client, robot_id, v[0], u[0])

        ## Change camera view location/angle
        position, orientation = get_robot_pose(pb_client, robot_id)
        set_camera(pb_client, 3, 0, -40, position)

        ## Compute MPC parameters
        initial_time_instance = k+1
        initial_pose = (position[0], position[1], orientation[2])
        x_initial = list(p_x)+list(p_y)+list(np.cos(theta))+list(np.sin(theta))+list(v)+list(u)

        ## Update debug text messages
        linear_velocity, angular_velocity = get_robot_velocity(pb_client, robot_id)
        # debug_text_id = pb_client.addUserDebugText('{:.2f}, {:.2f}, {:.2f}'.format(linear_velocity[0], linear_velocity[1], angular_velocity[2]),
        #                                            textPosition=(.1,0,.5),
        #                                            textColorRGB=(0,0,0),
        #                                            parentObjectUniqueId=robot_id,
        #                                            parentLinkIndex=0,
        #                                            replaceItemUniqueId=debug_text_id)

        for _ in range(12):
            pb.stepSimulation()
            # time.sleep(.01/240)


    # cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    # print(cubePos,cubeOrn)

    # pb_client.stopStateLogging(logging_unique_id)
    pb_client.disconnect()
