import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np

joint_indices = {'wheels':[2,3,5,7], 'steering_hinge': [4,6]}
wheel_radius = 0.05
car_length = 0.325
car_width = 2*0.1

## Fill out apply_action function for velocity control
## wheel_radius: wheel radius (r)
## car_width: width of the car (W)
## car_length: length of the car (L)
def apply_action(pb_client, body_id, linear_velocity, angular_velocity):

    wheel_velocity_rear_left = 0
    wheel_velocity_rear_right = 0
    wheel_velocity_front_left = 0
    wheel_velocity_front_right = 0
    steering_angle = 0
    
    if (np.abs(linear_velocity) > .0001):
        ##### EXERCISE #####
        ## Given the linear and angular velocities (in the robot frame), compute the steering angle and wheel velocities.
    
        steering_angle = np.arctan((angular_velocity*car_length)/linear_velocity)

        wheel_velocity_rear_left = (linear_velocity*(1 - (car_width/2*car_length)*np.tan(steering_angle)))/wheel_radius
        wheel_velocity_rear_right = (linear_velocity*(1 + (car_width/2*car_length)*np.tan(steering_angle)))/wheel_radius
        wheel_velocity_front_left = ((linear_velocity)/wheel_radius)*(np.cos(steering_angle) - (car_width/2*car_length)*np.sin(steering_angle) + np.sin(steering_angle)*np.tan(steering_angle))
        wheel_velocity_front_right = ((linear_velocity)/wheel_radius)*(np.cos(steering_angle) + (car_width/2*car_length)*np.sin(steering_angle) + np.sin(steering_angle)*np.tan(steering_angle))
        
        ##### /EXERCISE #####
    
        for joint_index in joint_indices['steering_hinge']:
            joint_position_control(pb_client, body_id, joint_index, steering_angle)

    wheel_velocities = [wheel_velocity_rear_left, wheel_velocity_rear_right, wheel_velocity_front_left, wheel_velocity_front_right]
    for i in range(4):
        joint_index = joint_indices['wheels'][i]
        wheel_velocity = wheel_velocities[i]

        joint_velocity_control(pb_client, body_id, joint_index, wheel_velocity)

def joint_position_control(pb_client, body_id, joint_index, value):
    pb_client.setJointMotorControl2(bodyUniqueId=body_id,
                                    jointIndex=joint_index,
                                    targetPosition=value,
                                    controlMode=pb.POSITION_CONTROL)

def joint_velocity_control(pb_client, body_id, joint_index, value):
    pb_client.setJointMotorControl2(bodyUniqueId=body_id,
                                    jointIndex=joint_index,
                                    targetVelocity=value,
                                    controlMode=pb.VELOCITY_CONTROL)
