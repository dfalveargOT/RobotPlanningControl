import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np

wheel_index = {'left':1, 'right':2}
wheel_radius = .061
wheel_baseline = 0.18738 # this is the distance between a wheel and the center of the robot

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

    pb_client.resetDebugVisualizerCamera(cameraDistance=2,
                                         cameraYaw=0,
                                         cameraPitch=-20,
                                         cameraTargetPosition=(0,0,.5))

    pb_client.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
    pb_client.setGravity(0,0,-10)

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

def draw_reference_trajectory(pb_client, object_id, trajectory, line_width=1):
    data_point = len(trajectory)

    for index in range(data_point-1):
        state = trajectory[index]
        next_state = trajectory[index+1]
        
        pb_client.addUserDebugLine(lineFromXYZ=(state[0], state[1],0),
                                   lineToXYZ=(next_state[0],next_state[1],0),
                                   lineColorRGB=(0,0,0),
                                   lineWidth=line_width,
                                   parentObjectUniqueId=object_id,
                                   parentLinkIndex=0)


def add_debug_parameters(pb_client, parameter_info):
    debug_parameter_ids = []
    for data in parameter_info:
        debug_parameter_id = pb_client.addUserDebugParameter(paramName=data['name'],
                                                             rangeMin=data['lower_limit'],
                                                             rangeMax=data['upper_limit'],
                                                             startValue=data['start_value'])
        
        debug_parameter_ids.append(debug_parameter_id)

    return debug_parameter_ids

# def add_joint_debug_parameters(pb_client, joint_info):
#     debug_parameter_ids = []
#     for joint_data in joint_info:
#         joint_index = joint_data['index']
#         joint_name = joint_data['name'].decode('utf-8')

#         if (joint_data['limit'][0] < joint_data['limit'][1]):
#             joint_lower_limit, joint_upper_limit = joint_data['limit']
            
#         else:
#             joint_lower_limit, joint_upper_limit = -10,10

#         start_value = (joint_lower_limit+joint_upper_limit)/2
        
#         if (joint_index == 3):
#             start_value = 0
            
#         elif (joint_index in [12,13,15,17]):
#             start_value = np.pi/2
            
#         debug_parameter_id = pb_client.addUserDebugParameter(paramName=joint_name,
#                                                              rangeMin=joint_lower_limit,
#                                                              rangeMax=joint_upper_limit,
#                                                              startValue=start_value)
        
#         debug_parameter_ids.append(debug_parameter_id)

#     return debug_parameter_ids

def get_robot_pose(pb_client, robot_id):
    position, quaternion = pb_client.getBasePositionAndOrientation(bodyUniqueId=robot_id)
    orientation = pb_client.getEulerFromQuaternion(quaternion)

    return position, orientation

def get_robot_velocity(pb_client, robot_id):
    linear_velocity, angular_velocity = pb_client.getBaseVelocity(bodyUniqueId=robot_id)

    return linear_velocity, angular_velocity

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

def apply_action(pb_client, robot_id, linear_velocity, angular_velocity):
    wheel_velocity_left = (linear_velocity + wheel_baseline*angular_velocity)/wheel_radius
    wheel_velocity_right = (linear_velocity - wheel_baseline*angular_velocity)/wheel_radius

    wheel_index_left = wheel_index['left']
    wheel_index_right = wheel_index['right']
    
    joint_velocity_control(pb_client, robot_id, wheel_index_left, wheel_velocity_left)
    joint_velocity_control(pb_client, robot_id, wheel_index_right, wheel_velocity_right)

def joint_position_control(pb_client, robot_id, joint_index, value):
    pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                    jointIndex=joint_index,
                                    targetPosition=value,
                                    maxVelocity=1,
                                    controlMode=pb.POSITION_CONTROL)

def joint_velocity_control(pb_client, robot_id, joint_index, value):
    pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                    jointIndex=joint_index,
                                    targetVelocity=value,
                                    controlMode=pb.VELOCITY_CONTROL,
                                    force=100000)

def lower_torso(pb_client, robot_id):
    joint_position_control(pb_client, robot_id, 3, 0)
    
def retract_arm(pb_client, robot_id):
    for joint_index in [12,13,15,17]:
        joint_position_control(pb_client, robot_id, joint_index, np.pi/2)
