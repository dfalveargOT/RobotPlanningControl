import numpy as np
import scipy.sparse as sps
from scipy.linalg import block_diag
import cyipopt

from shapely.geometry import Polygon

from BSplineOptimizationJAXEDT import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == '__main__':
    ## Trajectory planning parameters
    origin = (0,0)
    destination = (10, 10)
    obstacles = [] # list of obstacles' polygonal representation (points)
    obstacles.append(Polygon([[2,2],
                              [4,2],
                              [4,6],
                              [2,6],
                              [2,2]]))
    

    obstacles.append(Polygon([[4,6],
                              [8,6],
                              [8,8],
                              [4,8],
                              [4,6]]))
    

    bspline = BSplineOptimization(number_of_control_points=10, dt=.1)
    bspline.set_initial_state(position=origin, velocity=(.5,.1), acceleration=(0,0))
    bspline.set_final_state(position=destination, velocity=(.5,0), acceleration=(0,0))
    bspline.set_obstacles(obstacles)

    x_initial = 1*np.random.random(2*bspline.number_of_variables)
    p_x, p_y, control_points_x, control_points_y = bspline.compute_trajectory(x_initial)
    
    ## Plot the environment
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)

    ## Plot origin
    ax.plot(*origin, 'o')

    ## Plot destination
    ax.plot(*destination, 'x')

    ## Plot obstacles
    for obstacle in obstacles:
        plt.fill(*obstacle.exterior.xy, color='black')

    ax.axis('equal')
    ax.set_xlim([-1, 11])
    ax.set_ylim([-1, 11])

    plt.show()

    ## Plot results
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)

    ## Plot control points
    for i in range(bspline.number_of_variables):
        ax.plot(control_points_x[i], control_points_y[i], 'k.')
        
    ## Plot trajectory
    ax.plot(p_x, p_y)

    ## Plot origin
    ax.plot(*origin, 'o')

    ## Plot destination
    ax.plot(*destination, 'x')

    ## Plot obstacles
    for obstacle in obstacles:
        plt.fill(*obstacle.exterior.xy, color='black')

    ax.axis('equal')
    ax.set_xlim([-1, 11])
    ax.set_ylim([-1, 11])

    plt.show()
        

