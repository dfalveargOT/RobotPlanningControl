import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np

joint_indices = {'steering_hinge':0, 'wheels': [1,2,3]}
wheel_radius = 0.566
tricycle_length = 1.59
tricycle_width = 1.0

## Fill out apply_action function for velocity control
def apply_action_exercise(pb_client, body_id, linear_velocity, angular_velocity):
    ##### EXERCISE #####
    ## Given the desired linear and angular velocities of tricycle, compute appropriate wheel velocities
    wheel_velocity_front = 0
    wheel_velocity_left = 0
    wheel_velocity_right = 0

    steering_angle = 0

    ##### /EXERCISE #####

    wheel_velocities = [wheel_velocity_front, wheel_velocity_left, wheel_velocity_right]
    for i in range(3):
        joint_index = joint_indices['wheels'][i]
        wheel_velocity = wheel_velocities[i]

        joint_velocity_control(pb_client, body_id, joint_index, wheel_velocity)

    joint_index = joint_indices['steering_hinge']
    joint_position_control(pb_client, body_id, joint_index, steering_angle)

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
