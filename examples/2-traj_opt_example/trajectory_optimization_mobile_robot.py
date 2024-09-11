import numpy as np
import scipy.sparse as sps
from scipy.linalg import block_diag
import cyipopt

from TrajectoryOptimization import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == '__main__':
    ## Trajectory planning parameters
    origin = (0,0)
    destination = (10, 10)
    # obstacles = [] # list of obstacles' position and radius
    obstacles = [(4, 5, 1), (2, 2, 1), (7, 5, 1)] # list of obstacles' position and radius
    # obstacles = [(3, 3, 1), (5, 3, 1), (5, 5, 1)] # list of obstacles' position and radius

    traj_opt = TrajectoryOptimization(dt=.2, alpha=.01)
    traj_opt.set_origin(origin)
    traj_opt.set_destination(destination)
    traj_opt.set_obstacles(obstacles)

    x_initial = 1*np.random.random(2*traj_opt.N)
    p_x, p_y = traj_opt.compute_trajectory(x_initial)
    
    ## Plot the environment
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)

    ## Plot origin
    ax.plot(*origin, 'o')

    ## Plot destination
    ax.plot(*destination, 'x')

    ## Plot obstacles
    for obstacle in obstacles:
        obstacle_poly = plt.Circle(obstacle[:2], obstacle[2], color='r')
        ax.add_patch(obstacle_poly)

    ax.axis('equal')
    ax.set_xlim([-1, 11])
    ax.set_ylim([-1, 11])

    plt.show()

    ## Plot results
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
        
    ## Plot trajectory
    ax.plot(p_x, p_y)

    ## Plot origin
    ax.plot(*origin, 'o')

    ## Plot destination
    ax.plot(*destination, 'x')

    ## Plot obstacles
    for obstacle in obstacles:
        obstacle_poly = plt.Circle(obstacle[:2], obstacle[2], color='r')
        ax.add_patch(obstacle_poly)

    ax.axis('equal')
    ax.set_xlim([-1, 11])
    ax.set_ylim([-1, 11])

    plt.show()
        

