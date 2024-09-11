import control
import numpy as np
import math

from kuka_dynamics import *
from utils import *
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from copy import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Cube:
    """Represents a cube object in the simulation with tracking of position, trajectory and velocity.
    """
    def __init__(self, position, idx) -> None:
        """Initializes the Cube object.

        Args:
            position (tuple): The initial position of the cube.
            idx (int): The index of the cube.
        """
        self.last_position = position
        self.trajectory = []
        self.iter_time = 1/240
        self.idx = idx
        self.velocity = [0,0,0]
        self.time_prediction = 0.15 # seconds  0.38
        self.grasped = False
    
    def update_position(self, position):
        """Updates the position of the cube and appends it to the trajectory.

        Args:
            position (tuple): The new position of the cube.
        """
        self.trajectory.append(position)
        self.last_position = position
    
    def calculate_velocity(self):
        """Calculates the average velocity of the cube based on its trajectory."""
        # given the last positions of the cube calculate the velocity
        if len(self.trajectory) > 1:
            dist = []
            for i in range(len(self.trajectory) - 1):
                current = self.trajectory[i]
                last = self.trajectory[i+1]
                distance = self.calculate_distance(current, last)
                dist.append(distance)
            self.velocity = np.array(dist).mean(axis=0) / self.iter_time
            # print(self.velocity)
    
    def predict_position(self):
        """Predicts the future position of the cube based on its current position and velocity.

        Returns:
            tuple: The predicted position of the cube.
        """
        # predict the next position of the object 
        self.calculate_velocity()
        x, y, z = self.last_position
        vx, vy, vz = self.velocity
        t = self.time_prediction
        return (x - vx*t, y - vy*t, z + vz*t)
    
    def calculate_distance(self, pos1, pos2):
        """Calculates the distance between two positions.

        Args:
            pos1 (tuple): The first position.
            pos2 (tuple): The second position.

        Returns:
            list: A list with the distances in each axis [x, y, z].
        """
        # calculate the distance between two objects
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        return [x1 - x2, y1 - y2, z1 - z2]
        # return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


class Cubes:
    """Manages multiple Cube objects in the simulation, handling updates, creation, and removal.
    """
    def __init__(self, iter_time, same_cube_threshold=0.1) -> None:
        """Initializes the Cubes object.

        Args:
            iter_time (float): The iteration time for the simulation.
            same_cube_threshold (float, optional): The distance threshold to consider two cubes as the same. Defaults to 0.1.
        """
        self.iter_time = iter_time
        self.cubes = []
        self.same_cube_threshold = same_cube_threshold
        self.idx = 0
        self.belt_end_position_x = -0.3
        self.belt_end_position_y_robotside = -0.25
        self.belt_end_position_y_beltside = -0.8
    
    def update_positions(self, positions):
        """Updates the positions of the cubes in the simulation or creates new ones if necessary.

        Args:
            positions (list): A list of tuples representing the positions of the cubes.
        """
        # check if the cube is already created
        # if not create the object
        # self.remove_out_of_workspace()
        for pos in positions:
            updated_flag = False
            for cube in self.cubes:
                # get the last position of the cube
                pos_cube = cube.last_position 
                distance = self.calculate_distance(pos, pos_cube)
                # print(distance)
                if distance < self.same_cube_threshold:
                    # distance is less than threshold
                    # correspond to same cube
                    cube.update_position(pos)
                    updated_flag = True
                    break
            if updated_flag == False:
                # check if the cube is not in the belt
                self.create_cube(pos, self.idx)
                # print(f"created cube from total {len(self.cubes)}")
                self.idx += 1
    
    def remove_out_of_workspace(self):
        """Removes cubes that are outside of the belt's workspace."""
        init_val = len(self.cubes)
        for idx, cube in enumerate(self.cubes):
                # get the last position of the cube
                position = cube.last_position 
                if position[0] < self.belt_end_position_x or position[1] > self.belt_end_position_y_robotside or position[1] < self.belt_end_position_y_beltside:
                    # self.cubes.pop(idx)
                    
                    self.cubes.remove(cube)
        # print(f"init_val: {init_val}, final: {len(self.cubes)}")
        
    def create_cube(self, position, idx):
        """Creates a new Cube object with the given position and index.

        Args:
            position (tuple): The initial position of the cube.
            idx (int): The index of the cube.
        """
        # create the new cube object
        new_cube = Cube(position, idx)
        # update the current position of the object
        new_cube.update_position(position) 
        # add to the cube list
        self.cubes.append(new_cube)
    
    def calculate_distance(self, pos1, pos2):
        """Calculates the Euclidean distance between two positions.

        Args:
            pos1 (tuple): The first position.
            pos2 (tuple): The second position.

        Returns:
            float: The Euclidean distance between the two positions.
        """
        # calculate the distance between two objects
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2

        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        
class ManipulatorController:
    """
    This class implements the Linear Quadratic Regulator (LQR) control algorithm and computes torque control
    for a robotic manipulator. It also includes functions for joint angle control and inverse kinematics.
    """
    def __init__(self, pb_client, robot_id, arm_joint_indices):
        """
        Constructor for the ManipulatorController class.

        Args:
            pb_client: PyBullet client for the physics simulation.
            robot_id: The unique identifier of the robot in the simulation.
            arm_joint_indices: A list of joint indices for the robot's arm.
        """
        self.arm_joint_indices = arm_joint_indices
        self.pb_client = pb_client
        self.robot_id = robot_id
        
        ## Compute K_1 and K_2 for LQR control using control python package
        K_1 = np.zeros((7,7))
        K_2 = np.zeros((7,7))
        # Define state equation variables
        A1 = np.concatenate((np.zeros((7,7)), np.eye(7)), axis=1)
        A2 = np.concatenate((np.zeros((7,7)), np.zeros((7,7))), axis=1)

        self.A = np.concatenate((A1, A2), axis=0)
        self.B = np.matrix(np.concatenate((np.zeros((7,7)), np.eye(7))))
        self.Q = np.identity(14)

        for i in range(7):
            self.Q[i, i] = 800 #800
            self.Q[i + 7, i + 7] = 10 # 1
            
        # Modify the Q matrix diagonal values for end-effector joints
        for i in range(5, 7):  # Adjust the range if needed
            self.Q[i, i] = 2500  # Increase the value for joint angle error 2500
            self.Q[i + 7, i + 7] = 50  # Increase the value for joint velocity error 50

        self.R = 0.05*np.identity(7) # 0.1
        # compute lqr with control package
        print(f'Manipulator Controller: Comupte LQR control gain')
        self.compute_control_gain(self.A, self.B, self.Q, self.R)
    
    def compute_control_gain(self, A, B, Q, R):
        """
        Compute the control gain matrices K_1 and K_2 for the LQR control algorithm.

        Args:
            A: State-transition matrix.
            B: Control input matrix.
            Q: State cost matrix.
            R: Control input cost matrix.
        """
        control_gain, _, _ = control.lqr(A,B,Q,R)
        self.K_1 = -control_gain[:,:7] # get the K1 
        self.K_2 = -control_gain[:,7:] # get the K2
        # print(f'K1: {self.K_1}')
        # print(f'K2: {self.K_1}')
        
    def compute_torque_control(self, joint_angles, joint_velocities, q_ref, q_dot_ref):
        """
        Compute the torque control for the robotic manipulator given joint angles and joint velocities.

        Args:
            joint_angles: A NumPy array of joint angles.
            joint_velocities: A NumPy array of joint velocities.
            q_ref: A NumPy array of reference joint angles.
            q_dot_ref: A NumPy array of reference joint velocities.

        Returns:
            tau: A NumPy array of computed torques.
        """
        theta_ddot = self.K_1.dot(joint_angles - q_ref) + self.K_2.dot(joint_velocities - q_dot_ref)
        tau = iterative_ne_algorithm(len(self.arm_joint_indices), joint_angles, joint_velocities, theta_ddot)
        return tau
    
    def joint_angle_control(self, ee_pose, ee_orientation):
        """
        Calculate the desired joint angles given end-effector pose and orientation.

        Args:
            ee_pose: A list containing the end-effector position [x, y, z].
            ee_orientation: A list containing the end-effector orientation as Euler angles [roll, pitch, yaw].

        Returns:
            joint_angle_desired: A NumPy array of desired joint angles.
        """
        joint_angle_desired = inverse_kinematics(self.pb_client, self.robot_id, ee_pose, ee_orientation)
        return np.array(joint_angle_desired)
    
class TaskAllocator:
    """
    This class is responsible for allocating tasks to the robotic manipulator, such as picking up objects
    and placing them in specified trays. The class contains various utility functions for calculating distances,
    manipulating waypoints, and avoiding singularities.
    """
    def __init__(self, controller:ManipulatorController, trays, tray_width, h=1/240) -> None:
        """Constructor TaskAllocator.

        Args:
            controller (ManipulatorController): An instance of the ManipulatorController class.
            trays (list): A list containing the positions of trays.
            tray_width (float): The width of the trays.
            h (float): The time interval for the trajectory.
        """
        self.controller = controller # use the manipulator controller to compute the torques
        self.trays = trays # positions of the trays default(red, green, blue)
        self.tray_width = tray_width
        
        self.home_pose = [[0,-.5,1.1], [np.pi,0,0]]
        self.flag_obj_grasped = False
        self.flag_release = False
        self.grasped_waypoints = None
        self.grasped_obj = -1
        
        self.object_clearance = 0.10 # 0.12
        self.distance_manipulator = 0.8 # 1.1
        self.grasp_threshold = 0.38 # self.object_clearance
        self.pose_threshold = 0.20
        self.tray_clearance = 0.35
        self.orientation_threshold = 0.5
        self.time_tolerance = 150
        
        # task assigned
        self.previous_ee_position = None
        self.count_singul = 0
        
        # Smooth trajectory section
        # Define the total time T and the time interval h
        self.T = 10 # time estimated to complete the trajectory
        self.h = 0.5 
        
        # cubes 
        self.cubes = Cubes(h)
        
        # paths to each tray
        self.trays[0][2] += self.tray_clearance 
        self.trays[1][2] += self.tray_clearance 
        self.trays[2][2] += self.tray_clearance 
        self.trays[0][0] -= 0.1 # update for red in x
        self.trays[0][1] += 0.1 # update for red in x
        # self.trays[2][0] += self.tray_clearance # update for blue in x
        self.path_tray_1 = [self.home_pose[0], [-0.5,-0.25,0.94], [-0.7,0.3,0.92], self.trays[0], [-0.7,0.31,0.93], [-0.51,-0.26,0.91]]
        self.path_tray_2 = [self.home_pose[0], [-0.5,-0.25,1.2], [-0.7,0.3,1.2], self.trays[1], [-0.7,0.3,0.9], [-0.5,-0.25,0.9]]
        self.path_tray_3 = [self.home_pose[0], [0.4,-0.25,1.1], [0.5,0.3,1.1], [0.75,0.5,0.9], self.trays[2], [0.75,0.5,0.9],  [0.5,0.3,1.1], [0.4,-0.25,1.1]]
        self.paths = [self.path_tray_1, self.path_tray_2, self.path_tray_3]
        
        # optional paths for the trajectories
        self.optpath_tray_1 = [self.home_pose[0], [0.4,-0.25,1.1], [0.5,0.3,1.1], [0.75,0.5,0.9], self.trays[1], self.trays[0], self.trays[1], [0.75,0.5,0.9],  [0.5,0.3,1.1], [0.4,-0.25,1.1]]
        self.optpath_tray_2 = [self.home_pose[0], [0.4,-0.25,1.1], [0.5,0.3,1.1], [0.75,0.5,0.9], self.trays[1], [0.75,0.5,0.9],  [0.5,0.3,1.1], [0.4,-0.25,1.1]]
        self.optpath_tray_3 = [self.home_pose[0], [-0.5,-0.25,1.1], [-0.7,0.3,1.1], [0.25,0.6,0.9], self.trays[2], [0,0.4,1.1], [-0.7,0.3,1.1], [-0.5,-0.25,1.1]]
        self.optpaths = [self.optpath_tray_1, self.optpath_tray_2, self.optpath_tray_3]
        
    def allocate(self, objects, joint_angles, joint_velocities, end_effector_pose):
        """
        Determines the appropriate torque and joint angle commands for the manipulator
        based on the current state and environment.

        Args:
            objects (list): A list of objects with their positions.
            joint_angles (numpy.ndarray): The current joint angles of the manipulator.
            joint_velocities (numpy.ndarray): The current joint velocities of the manipulator.
            end_effector_pose (list): The current pose of the end effector.

        Returns:
            torques (numpy.ndarray): The computed torque commands for the manipulator.
            q_ref (numpy.ndarray): The computed joint angle commands for the manipulator.
        """
        # always allocate first object
        # print(end_effector_pose)
        self.cubes.remove_out_of_workspace()
        self.cubes.update_positions(objects)
        if self.flag_obj_grasped == False:
            
            if len(objects) > 0:
                # obj_pose_1 = objects[0]
                cube = self.cubes.cubes[0]
                obj_pose = cube.predict_position()
                obj_pose_last = cube.last_position
                # print(f'original pose: {obj_pose_1}, predicted: {obj_pose}')
                torques, q_ref = self.send_to_pose(joint_angles, joint_velocities, obj_pose)
                distance = self.calculate_distance(obj_pose_last, end_effector_pose[0])
                # error_orientation = self.calculate_orientation_error(self.home_pose[1], end_effector_pose[1])
                
                if distance < self.grasp_threshold:# and error_orientation < self.orientation_threshold:
                    # print("Near to object. Performing action.") 
                    obj_grasped = grasp_object(self.controller.pb_client, self.controller.robot_id)
                    if obj_grasped != -1 and self.flag_obj_grasped == False:
                        self.flag_obj_grasped = True
                        self.grasped_obj = obj_grasped
                        self.grasped_waypoints = copy(self.paths[obj_grasped[1]])
                        print(f"1. Picked up: {obj_grasped}")
                        # Delete cube from the pool
                        self.cubes.cubes.remove(cube)
                        return self.send_to_home(joint_angles, joint_velocities)
                
                        
                elif distance > self.distance_manipulator:# or error_orientation > self.orientation_threshold:
                    return self.send_to_home(joint_angles, joint_velocities)
            else:
                torques, q_ref = self.send_to_home(joint_angles, joint_velocities)
        
        else:
            if len(self.grasped_waypoints) < 1 and self.flag_release:
                print("3. Return to next \n")
                self.flag_obj_grasped = False
                self.flag_release = False
                self.grasped_obj = -1
                self.grasped_waypoints = None
                return self.send_to_home(joint_angles, joint_velocities)
            

            
            # calculate distance to waypoint  
            waypoint = self.grasped_waypoints[0]
            distance = self.calculate_distance(waypoint, end_effector_pose[0])
            torques, q_ref = self.send_to_pose(joint_angles, joint_velocities, waypoint)
            # check if the end effector is near to the waypoint and pass the next one
            # print(f"distance {distance}")
            if distance < self.pose_threshold:
                # del self.grasped_waypoints[0] # eliminate the achieved waypoint
                if waypoint[:2] == self.trays[self.grasped_obj[1]][:2]: # check if the waypoint is a tray
                    # release the object and go for the next
                    print("2. release object")
                    relase_object(self.controller.pb_client)
                    self.flag_release = True
                
                del self.grasped_waypoints[0] # eliminate the achieved waypoint
            
            # check if the arm get stuck
            flag = self.checking_singuarity(end_effector_pose[0])
            if flag:
                self.grasped_waypoints = copy(self.optpaths[self.grasped_obj[1]])
                
                    
        
        return torques, q_ref
        
    def send_to_home(self, joint_angles, joint_velocities):
        """
        Computes the torque and joint angle commands to send the manipulator to its home pose.

        Args:
            joint_angles (numpy.ndarray): The current joint angles of the manipulator.
            joint_velocities (numpy.ndarray): The current joint velocities of the manipulator.

        Returns:
            torques (numpy.ndarray): The computed torque commands for the manipulator.
            q_ref (numpy.ndarray): The computed joint angle commands for the manipulator.
        """
        # calculate the joint angles desired
        q_ref = self.controller.joint_angle_control(self.home_pose[0], self.home_pose[1])
        q_dot_ref = np.zeros_like(q_ref)
        # send to calculate the torque control
        torques = self.controller.compute_torque_control(joint_angles, joint_velocities, q_ref, q_dot_ref)

        # return torques and joint_angles
        return torques, q_ref
    
    def send_to_pose(self, joint_angles, joint_velocities, pose):
        """
        Computes the torque and joint angle commands to send the manipulator to a target pose.

        Args:
            joint_angles (numpy.ndarray): The current joint angles of the manipulator.
            joint_velocities (numpy.ndarray): The current joint velocities of the manipulator.
        pose (list): The target pose for the manipulator (position and orientation).

        Returns:
            torques (numpy.ndarray): The computed torque commands for the manipulator.
            q_ref (numpy.ndarray): The computed joint angle commands for the manipulator.
        """
        # calculate the joint angles desired
        pose = [pose[0], pose[1], pose[2]+self.object_clearance]
        q_ref = self.controller.joint_angle_control(pose, self.home_pose[1]) # always maintain the pose of end effector
        q_dot_ref = np.zeros_like(q_ref)
        # send to calculate the torque control
        torques = self.controller.compute_torque_control(joint_angles, joint_velocities, q_ref, q_dot_ref)

        # return torques and joint_angles
        return torques, q_ref

    def calculate_distance(self, obj_pose, ee_pose):
        """
        Calculate the Euclidean distance between the end effector and the object pose.

        Parameters:
        obj_pose (tuple): (x, y, z) coordinates of the object pose.
        ee_pose (tuple): (x, y, z) coordinates of the end effector pose.

        Returns:
        float: Euclidean distance between the object and end effector poses.
        """
        """Calculate the distance between the end effector and the pose"""
        x1, y1, z1 = obj_pose
        x2, y2, z2 = ee_pose

        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        
        return distance
    
    def build_path(self, grasp_obj):
        """
        Create the waypoints to carry the object to the trays.

        Parameters:
        grasp_obj (tuple): A tuple containing object information.

        Returns:
        list: A list of waypoints to guide the manipulator from its home position to the target tray.
        """
        """Create the waypoints to carry the object to the trays"""
        way1 = self.home_pose[0]
        tray = self.trays[grasp_obj[1]]
        # way2 = [self.trays[1][0], self.trays[1][1], self.trays[1][2]+self.tray_clearance]
        way2 = [tray[0], tray[1], tray[2]+self.tray_clearance*1]
        # way3 = [tray[0], tray[1], tray[2]+self.tray_clearance]
        return [way1, way2]
    
    def calculate_orientation_error(self, desired, quaternion):
        """
        Calculate the orientation error between the desired and current end effector orientations.

        Parameters:
        desired (tuple): Desired (x, y, z) Euler angles.
        quaternion (tuple): Current end effector orientation in quaternion form.

        Returns:
        float: Euclidean distance representing the orientation error.
        """
        r = Rotation.from_quat(quaternion)
        ee_rot = r.as_euler('xyz', degrees=False)
        # calculate the error in orientation
        # print(f'desired: {desired}, current: {ee_rot}')
        
        return self.calculate_distance(desired, ee_rot)
    
    def checking_singuarity(self, ee_position):
        """
        Check if the manipulator is in a singular configuration or if it is stuck.

        Parameters:
        ee_position (tuple): (x, y, z) coordinates of the end effector position.

        Returns:
        bool: True if the manipulator is in a singular configuration or stuck, False otherwise.
        """
        flag = False
        if type(self.previous_ee_position) == np.ndarray:
            distance = self.calculate_distance(self.previous_ee_position, ee_position)
            self.previous_ee_position = ee_position
            # print(distance)
            if distance < 0.001:
                self.count_singul += 1
        else:
            self.previous_ee_position = ee_position
        
        # check if the arm is stuck
        if self.count_singul > self.time_tolerance:
            flag = True
            self.count_singul = 0
            print("Singularity")

        return flag