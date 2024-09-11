import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np

from kuka_dynamics import *
from kuka_kinematics import *
import control

robot_id = -1
arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
end_effector_link_index = 7
constraint_id = -1
grasped_cube_id = -1
cube_indices = []
cube_colors = {}

def set_camera(distance, yaw, pitch, position):
    pb.resetDebugVisualizerCamera(cameraDistance=distance,
                                  cameraYaw=yaw,
                                  cameraPitch=pitch,
                                  cameraTargetPosition=position)

def initializeGUI(enable_gui=True, gravity=-10):
    pb_client = bc.BulletClient(connection_mode=pb.GUI)
    
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, False)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_GUI, enable_gui)

    pb_client.resetDebugVisualizerCamera(cameraDistance=2,
                                         cameraYaw=0,
                                         cameraPitch=-20,
                                         cameraTargetPosition=(0,0,.5))

    pb_client.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
    pb_client.setGravity(0,0,gravity)

    return pb_client

def draw_frame(pb_client, robot_id, link_index, axis_length=.2, line_width=1):
    pb_client.addUserDebugLine(lineFromXYZ=(0,0,0),
                               lineToXYZ=(axis_length,0,0),
                               lineColorRGB=(1,0,0),
                               lineWidth=line_width,
                               parentObjectUniqueId=robot_id,
                               parentLinkIndex=link_index)

    pb_client.addUserDebugLine(lineFromXYZ=(0,0,0),
                               lineToXYZ=(0,axis_length,0),
                               lineColorRGB=(0,1,0),
                               lineWidth=line_width,
                               parentObjectUniqueId=robot_id,
                               parentLinkIndex=link_index)

    pb_client.addUserDebugLine(lineFromXYZ=(0,0,0),
                               lineToXYZ=(0,0,axis_length),
                               lineColorRGB=(0,0,1),
                               lineWidth=line_width,
                               parentObjectUniqueId=robot_id,
                               parentLinkIndex=link_index)
    
def mark_point(pb_client, point, axis_length=.2, line_width=1):
    point = np.array(point)
    
    pb_client.addUserDebugLine(lineFromXYZ=point,
                               lineToXYZ=point+(axis_length,0,0),
                               lineWidth=line_width,
                               lineColorRGB=(1,0,0))

    pb_client.addUserDebugLine(lineFromXYZ=point,
                               lineToXYZ=point+(0,axis_length,0),
                               lineWidth=line_width,
                               lineColorRGB=(0,1,0))

    pb_client.addUserDebugLine(lineFromXYZ=point,
                               lineToXYZ=point+(0,0,axis_length),
                               lineWidth=line_width,
                               lineColorRGB=(0,0,1))
    
def add_debug_parameters(pb_client, parameter_info):
    debug_parameter_ids = []
    for data in parameter_info:
        debug_parameter_id = pb_client.addUserDebugParameter(paramName=data['name'],
                                                             rangeMin=data['lower_limit'],
                                                             rangeMax=data['upper_limit'],
                                                             startValue=data['start_value'])
        
        debug_parameter_ids.append(debug_parameter_id)

    return debug_parameter_ids

def add_joint_debug_parameters(pb_client, arm_joint_indices, gripper_joint_indices=[]):
    debug_parameter_ids = []

    for joint_index in arm_joint_indices:
        joint_name = 'arm joint {}'.format(joint_index+1)
        joint_lower_limit = -180
        joint_upper_limit = 180
        start_value = (joint_lower_limit+joint_upper_limit)/2
        
        debug_parameter_id = pb_client.addUserDebugParameter(paramName=joint_name,
                                                             rangeMin=joint_lower_limit,
                                                             rangeMax=joint_upper_limit,
                                                             startValue=start_value)
        
        debug_parameter_ids.append(debug_parameter_id)

    for joint_index in gripper_joint_indices:
        joint_name = 'gripper joint {}'.format(joint_index+1)
        joint_lower_limit = -180
        joint_upper_limit = 180
        start_value = (joint_lower_limit+joint_upper_limit)/2
        
        debug_parameter_id = pb_client.addUserDebugParameter(paramName=joint_name,
                                                             rangeMin=joint_lower_limit,
                                                             rangeMax=joint_upper_limit,
                                                             startValue=start_value)
        
        debug_parameter_ids.append(debug_parameter_id)

    return debug_parameter_ids

def inverse_kinematics(pb_client, robot_id, target_position, target_orientation):
    target_quaternion = pb.getQuaternionFromEuler(target_orientation)
    joint_values = pb_client.calculateInverseKinematics(bodyUniqueId=robot_id,
                                                        endEffectorLinkIndex=end_effector_link_index,
                                                        targetPosition=target_position,
                                                        targetOrientation=target_quaternion)

    return joint_values

def get_jacobian(pb_client, robot_id, link_id, joint_indices):
    joint_angles = get_joint_angles(pb_client, robot_id, joint_indices)
    linear_jacobian, angular_jacobian = pb_client.calculateJacobian(bodyUniqueId=robot_id,
                                                                    linkIndex=link_id,
                                                                    localPosition=(0,0,0),
                                                                    objPositions=joint_angles,
                                                                    objVelocities=[0]*len(joint_angles),
                                                                    objAccelerations=[0]*len(joint_angles))

    return linear_jacobian, angular_jacobian

        
def get_end_effector_pose(pb_client, robot_id):
    link_world_position, link_world_orientation, _, _, _, _ = pb_client.getLinkState(bodyUniqueId=robot_id,
                                                                                     linkIndex=end_effector_link_index)

    return np.array(link_world_position), np.array(link_world_orientation)

def place_object(pb_client, location, color_code):
    global cube_indices, cube_colors

    if (color_code == 0):
        cube_index = pb_client.loadURDF("./robot_models/cube_red.urdf",
                                        location,
                                        globalScaling=3)
    elif (color_code == 1):
        cube_index = pb_client.loadURDF("./robot_models/cube_green.urdf",
                                        location,
                                        globalScaling=3)
    elif (color_code == 2):
        cube_index = pb_client.loadURDF("./robot_models/cube_blue.urdf",
                                        location,
                                        globalScaling=3)

    cube_indices.append(cube_index)
    cube_colors[cube_index] = color_code

def grasp_object(pb_client, robot_id):
    global constraint_id, grasped_cube_id

    if (constraint_id >= 0):
        return (grasped_cube_id, cube_colors[grasped_cube_id])

    for cube_index in cube_indices:
        contact_points = pb_client.getContactPoints(bodyA = robot_id,
                                                    bodyB = cube_index,
                                                    linkIndexA=end_effector_link_index,
                                                    linkIndexB=-1)

        if (len(contact_points) > 0):
            contact_point = contact_points[0]
            contact_normal = contact_point[7]
            contact_distance = contact_point[8]
            contact_force = contact_point[9]

            if (np.abs(contact_normal[2]) > 0.9 and contact_distance < .01 and contact_force > 10):
                constraint_id = pb_client.createConstraint(parentBodyUniqueId=robot_id,
                                                           parentLinkIndex=end_effector_link_index,
                                                           childBodyUniqueId=cube_index,
                                                           childLinkIndex=-1,
                                                           jointType=pb.JOINT_FIXED,
                                                           jointAxis=[0,1,0],
                                                           parentFramePosition=[0, 0, .1],
                                                           childFramePosition=[0, 0 ,0])

                grasped_cube_id = cube_index

                return (grasped_cube_id, cube_colors[grasped_cube_id])

    return -1        

def relase_object(pb_client):
    global constraint_id, grasped_cube_id

    if (constraint_id >= 0):
        pb_client.removeConstraint(constraint_id)
        constraint_id = -1
        grasped_cube_id = -1

def get_joint_info(pb_client, robot_id):
    number_of_joints = pb_client.getNumJoints(bodyUniqueId=robot_id)
    joint_info = []
    for joint_index in range(number_of_joints):
        return_data = pb_client.getJointInfo(bodyUniqueId=robot_id,
                                             jointIndex=joint_index)

    
        joint_index, joint_name = return_data[:2]
        joint_lower_limit = return_data[8]
        joint_upper_limit = return_data[9]
        joint_info.append({'index': joint_index,
                           'name': joint_name,
                           'limit': (joint_lower_limit, joint_upper_limit)})

    return joint_info

def get_joint_angles(pb_client, robot_id):
    joint_angles = []
    for joint_index in arm_joint_indices:
        position, velocity, force, torque = pb_client.getJointState(bodyUniqueId=robot_id,
                                                                    jointIndex=joint_index)

        joint_angles.append(position)

    return joint_angles

def get_joint_velocities(pb_client, robot_id):
    joint_velocities = []
    for joint_index in arm_joint_indices:
        position, velocity, force, torque = pb_client.getJointState(bodyUniqueId=robot_id,
                                                                    jointIndex=joint_index)

        joint_velocities.append(velocity)

    return joint_velocities

def free_joint_torques(pb_client, robot_id, joint_indices):
    pb_client.setJointMotorControlArray(bodyUniqueId=robot_id,
                                        jointIndices=joint_indices,
                                        controlMode=pb.VELOCITY_CONTROL,
                                        forces=np.zeros_like(joint_indices))

    
def joint_angle_control(pb_client, robot_id, joint_index, value):
    pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                    jointIndex=joint_index,
                                    targetPosition=value,
                                    controlMode=pb.POSITION_CONTROL)

    return

def joint_velocity_control(pb_client, robot_id, joint_index, value):
    pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                    jointIndex=joint_index,
                                    targetVelocity=value,
                                    controlMode=pb.VELOCITY_CONTROL)

    return

def joint_torque_control(pb_client, robot_id, joint_index, value):
    pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                    jointIndex=joint_index,
                                    force=value,
                                    controlMode=pb.TORQUE_CONTROL)

def get_cube_locations(pb_client):
    cube_locations = []
    for cube_index in cube_indices:    
        position, orientation = pb_client.getBasePositionAndOrientation(cube_index)

        if (position[0] < 1+.5 and position[0] > -1+.5 and
            position[1] < -.25+.03 and position[1] > -.25-.82-0.03 and
            position[2] > .5):

            cube_locations.append(position)

    return cube_locations
        
