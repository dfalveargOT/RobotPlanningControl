import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np

from kuka_dynamics import *
import control

arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]

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
        
def get_end_effector_pose(pb_client, robot_id, end_effector_link_index):
    link_world_position, link_world_orientation, _, _, _, _ = pb_client.getLinkState(bodyUniqueId=robot_id,
                                                                                     linkIndex=end_effector_link_index)

    return np.array(link_world_position), np.array(link_world_orientation)

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

def get_joint_angles(pb_client, robot_id, arm_joint_indices):
    joint_angles = []
    for joint_index in arm_joint_indices:
        position, velocity, force, torque = pb_client.getJointState(bodyUniqueId=robot_id,
                                                                    jointIndex=joint_index)

        joint_angles.append(position)

    return joint_angles

def free_joint_torques(pb_client, robot_id, joint_indices):
    pb_client.setJointMotorControlArray(bodyUniqueId=robot_id,
                                        jointIndices=joint_indices,
                                        controlMode=pb.VELOCITY_CONTROL,
                                        forces=np.zeros_like(joint_indices))

def joint_torque_control(pb_client, robot_id, joint_index, value):
    pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                    jointIndex=joint_index,
                                    force=value,
                                    controlMode=pb.TORQUE_CONTROL)
