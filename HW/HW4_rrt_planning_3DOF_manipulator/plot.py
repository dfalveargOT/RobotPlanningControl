import pdb, pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import transforms

number_of_joints = 3
link_length = 1
    
if (__name__ == "__main__"):
    with open('rrt_trajectory.pickle', 'rb') as filename:
        trajectory = pickle.load(filename)

    ## Plot robot trajectory
    for joint_angles in trajectory:
        
        joint_positions = [(0,0)]
        joint_angle_cumulative = 0
        for i in range(number_of_joints):
            previous_joint_position = joint_positions[-1]
            joint_angle_cumulative += joint_angles[i]
        
            joint_position_x = previous_joint_position[0] + link_length*np.cos(joint_angle_cumulative)
            joint_position_y = previous_joint_position[1] + link_length*np.sin(joint_angle_cumulative)

            joint_positions.append((joint_position_x, joint_position_y))
        
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
            
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(90)

        ## Draw manipulator
        joint_positions_x, joint_positions_y = zip(*joint_positions)
        plt.plot(joint_positions_x, joint_positions_y, transform=rot+base, solid_capstyle='round', linewidth=7)
        
        ## Draw joints
        for joint_position in joint_positions:
            plt.plot(joint_position[0], joint_position[1], marker='o', markersize=5, color='black', transform=rot+base)

        ## Draw end-effector
        plt.plot(joint_position[0], joint_position[1], marker='o', markersize=5, color='green', transform=rot+base)

        ## Draw obstacles
        rectangle = plt.Rectangle((.25, 1), .5, .5, fc='orange', ec='red', linewidth=5, transform=rot+base)
        plt.gca().add_patch(rectangle)
        rectangle = plt.Rectangle((1.5, .5), .5, .5, fc='orange', ec='red', linewidth=5, transform=rot+base)
        plt.gca().add_patch(rectangle)
        rectangle = plt.Rectangle((1, 2), .5, .5, fc='orange', ec='red', linewidth=5, transform=rot+base)
        plt.gca().add_patch(rectangle)

        ## Draw end-effector destination
        rectangle = plt.Rectangle((0.3677, 2.123), .1, .1, fc='blue', ec='green', linewidth=5, transform=rot+base)
        plt.gca().add_patch(rectangle)

        ax.axis('equal')
        ax.axis('off')
        ax.set_xlim([-3.2, 1.2])
        ax.set_ylim([-1.2, 3.2])

        plt.show()
