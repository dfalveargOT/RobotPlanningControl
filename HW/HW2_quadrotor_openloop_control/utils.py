import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np

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
    pb_client.setGravity(0, 0, gravity)

    return pb_client

def mark_waypoint(pb_client, waypoint, axis_length=.2, line_width=1):
    waypoint = np.array(waypoint)
    
    pb_client.addUserDebugLine(lineFromXYZ=waypoint,
                               lineToXYZ=waypoint+(axis_length,0,0),
                               lineWidth=line_width,
                               lineColorRGB=(1,0,0))

    pb_client.addUserDebugLine(lineFromXYZ=waypoint,
                               lineToXYZ=waypoint+(0,axis_length,0),
                               lineWidth=line_width,
                               lineColorRGB=(0,1,0))

    pb_client.addUserDebugLine(lineFromXYZ=waypoint,
                               lineToXYZ=waypoint+(0,0,axis_length),
                               lineWidth=line_width,
                               lineColorRGB=(0,0,1))
    
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

def get_pose(pb_client, robot_id):
    position, orientation = pb_client.getBasePositionAndOrientation(bodyUniqueId=robot_id)
    orientation = pb.getEulerFromQuaternion(orientation)

    return (np.array(position), np.array(orientation))

def get_velocity(pb_client, robot_id):
    position, orientation = get_pose(pb_client, robot_id)
    linear_velocity, angular_velocity = pb_client.getBaseVelocity(bodyUniqueId=robot_id)

    phi, theta, psi = orientation
    rotation_matrix = np.matrix([[np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi)],
                                 [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi)],
                                 [-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]])
    linear_velocity_local_frame = np.squeeze(np.asarray(np.inner(rotation_matrix.T, linear_velocity)))

    T_matrix = np.matrix([[1, 0, -np.sin(theta)],
                          [0, np.cos(phi), np.sin(phi)*np.cos(theta)],
                          [0, -np.sin(phi), np.cos(phi)*np.cos(theta)]])

    angular_velocity_local_frame = np.squeeze(np.asarray(np.inner(T_matrix, angular_velocity)))

    return (linear_velocity_local_frame, angular_velocity_local_frame)

def get_robot_state(pb_client, robot_id):
    return get_pose(pb_client, robot_id) + get_velocity(pb_client, robot_id)

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

def force_torque_control(pb_client, robot_id, control_input):
    pb_client.applyExternalForce(robot_id,
                                 -1,
                                 forceObj=[0,0,control_input[0]],
                                 posObj=(0,0,0),
                                 flags=pb.LINK_FRAME)

    pb_client.applyExternalTorque(robot_id,
                                  -1,
                                  torqueObj=control_input[1:],
                                  flags=pb.LINK_FRAME)

    


