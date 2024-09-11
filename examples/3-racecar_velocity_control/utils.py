import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np

joint_indices = {'wheels':[2,3,5,7], 'steering_hinge': [4,6]}
wheel_radius = 0.05
car_length = 0.325
car_width = 2*0.1

def set_camera(pb_client, distance, yaw, pitch, position):
    pb_client.resetDebugVisualizerCamera(cameraDistance=distance,
                                         cameraYaw=yaw,
                                         cameraPitch=pitch,
                                         cameraTargetPosition=position)

def initializeGUI(enable_gui=True):
    pb_client = bc.BulletClient(connection_mode=pb.GUI)
    
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, False)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_GUI, enable_gui)
    pb_client.configureDebugVisualizer(pb.COV_ENABLE_GUI, enable_gui)

    pb_client.resetDebugVisualizerCamera(cameraDistance=5,
                                  cameraYaw=40,
                                  cameraPitch=-30,
                                  cameraTargetPosition=(0,0,0))

    pb_client.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
    pb_client.setGravity(0,0,-10)

    return pb_client

def get_robot_pose(pb_client, robot_id):
    position, quaternion = pb_client.getBasePositionAndOrientation(bodyUniqueId=robot_id)
    orientation = pb_client.getEulerFromQuaternion(quaternion)

    return position, orientation

def get_robot_velocity(pb_client, robot_id):
    linear_velocity, angular_velocity = pb_client.getBaseVelocity(bodyUniqueId=robot_id)

    return linear_velocity, angular_velocity

def get_joint_info(pb_client, body_id):
    number_of_joints = pb_client.getNumJoints(bodyUniqueId=body_id)
    joint_info = []
    for joint_index in range(number_of_joints):
        return_data = pb_client.getJointInfo(bodyUniqueId=body_id,
                                      jointIndex=joint_index)

        joint_index, joint_name = return_data[:2]
        joint_lower_limit = return_data[8]
        joint_upper_limit = return_data[9]
        joint_info.append({'index': joint_index,
                           'name': joint_name,
                           'limit': (joint_lower_limit, joint_upper_limit)})

    return joint_info

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

def add_debug_parameters(pb_client, parameter_info):
    debug_parameter_ids = []
    for data in parameter_info:
        debug_parameter_id = pb_client.addUserDebugParameter(paramName=data['name'],
                                                             rangeMin=data['lower_limit'],
                                                             rangeMax=data['upper_limit'],
                                                             startValue=data['start_value'])
        
        debug_parameter_ids.append(debug_parameter_id)

    return debug_parameter_ids

def add_joint_debug_parameters(pb_client, joint_info):
    debug_parameter_ids = []

    for key in joint_indices:
        joint_lower_limit = 0
        joint_upper_limit = 0
        start_value = 0
        
        if (key == 'wheels'):
            joint_upper_limit = 2
            
        elif (key == 'steering_hinge'):
            joint_lower_limit = -np.pi/3
            joint_upper_limit = np.pi/3

        debug_parameter_id = pb_client.addUserDebugParameter(paramName=key,
                                                             rangeMin=joint_lower_limit,
                                                             rangeMax=joint_upper_limit,
                                                             startValue=start_value)
        
        debug_parameter_ids.append(debug_parameter_id)

    return debug_parameter_ids
