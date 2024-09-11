import sys, time, pdb, argparse, pickle
import numpy as np
from numpy.linalg import norm

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

## Validity checker for path planning
## Return whether the given joint angles is collision free
class ValidityChecker(ob.StateValidityChecker):
    number_of_joints = 3
    link_length = 1
        
    def isValid(self, state):
        number_of_joints = self.number_of_joints
        link_length = self.link_length
    
        joint_angles = [state[i].value for i in range(self.number_of_joints)]

        ########## EXERCISE ##########
        ## Check whether the given state (joint angles) of the 3DOF manipulator contacting with obstacles. 
        ## Use the following parameters/variables
        ## joint_angles: joing angle of the 3DOF manipulator
        ## number_of_joints: number of joints
        ## link_length: the length of each link
        
        # Translate joint angles to cartesian positions
        x = y = 0
        joint_angle = 0
        joint_positions = []
        for angle in joint_angles:
            joint_angle += angle
            x += link_length*np.cos(joint_angle) # position joint angle x coord
            y += link_length*np.sin(joint_angle) # position joint angle y coord
            joint_positions.append([x,y])
            
        # add midpositions
        # x_prev = y_prev = 0
        # mid_positions = []
        # for joint_pos in joint_positions:
        #     mid_x, mid_y = 0.5*(joint_pos[0] - x_prev), 0.5*(joint_pos[1] - y_prev)
        #     x_prev, y_prev = joint_pos
        #     mid_positions.append([mid_x,mid_y])
        
        # joint_positions += mid_positions
        
        # obstacles
        # pos_obstacles = np.array([[1.0,2.0],[0.25,1.0],[1.50,0.50]])
        center_obstacles = np.array([[1.25,2.25],[0.5,1.25],[1.50,0.50]])
        # center_obstacles = np.array([[1.25,2.25],[0.5,1.25],[1.75,0.75]])
        width_obstacles = 0.5
        radius = width_obstacles*0.5 + 0.05
        
        is_robot_not_contacting_obstacles = True
        
        # check intersection between points and obstacles
        # using a radius for each obstacle
        for obs in center_obstacles:
            for i in range(len(joint_positions) - 1):
                # distance = (np.linalg.norm(np.array(joint_positions[i]) - np.array(obs))) #<= 0.01 or distance <= radius
                if self.check_intersection(obs, joint_positions[i], joint_positions[i+1], radius):#or distance <= radius:
                    is_robot_not_contacting_obstacles = False
        
        # print("\n \n \n") print(f"obstacle: {distance}")
        return is_robot_not_contacting_obstacles

        ########## /EXERCISE ##########
    
    def check_intersection(self, obs_position, joint_pos1, joint_pos2, radius):
        
        # Calculate the direction vector from each joint
        direction = np.array(joint_pos2) - np.array(joint_pos1)
        direction /= np.linalg.norm(direction)
        
        # Calculate nearest point to the center
        near_point = np.array(obs_position) - np.array(joint_pos1)
        
        # project 
        proj = near_point.dot(direction)
        closest_point = np.array(joint_pos1) + proj * direction
        
        # calculate the distance between the closest point to the center
        distance = np.linalg.norm(closest_point - obs_position)**2
        
        # check if the distance is within the range
        if distance <= radius**2:
            # exist collision
            # check if the projection is in the length range
            if proj >= 0 and proj <= np.linalg.norm(np.array(joint_pos2) - np.array(joint_pos1)):
                return True
        
        return False
         

## Returns a structure representing the optimization objective to use
#  for optimal motion planning. This method returns an objective
#  which attempts to minimize the length in configuration space of
#  computed paths.
def getPathLengthObjective(si):
    return ob.PathLengthOptimizationObjective(si)

## Returns an optimization objective which attempts to minimize path
#  length that is satisfied when a path of length shorter than 1.51
#  is found.
def getThresholdPathLengthObj(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(1.51))
    return obj

## Defines an optimization objective which attempts to steer the
#  robot away from obstacles. To formulate this objective as a
#  minimization of path cost, we can define the cost of a path as a
#  summation of the costs of each of the states along the path, where
#  each state cost is a function of that state's clearance from
#  obstacles.
#
#  The class StateCostIntegralObjective represents objectives as
#  summations of state costs, just like we require. All we need to do
#  then is inherit from that base class and define our specific state
#  cost function by overriding the stateCost() method.
#
class ClearanceObjective(ob.StateCostIntegralObjective):
    def __init__(self, si):
        super(ClearanceObjective, self).__init__(si, True)
        self.si_ = si

    # Our requirement is to maximize path clearance from obstacles,
    # but we want to represent the objective as a path cost
    # minimization. Therefore, we set each state's cost to be the
    # reciprocal of its clearance, so that as state clearance
    # increases, the state cost decreases.
    def stateCost(self, s):
        return ob.Cost(1 / (self.si_.getStateValidityChecker().clearance(s) +
                            sys.float_info.min))

## Return an optimization objective which attempts to steer the robot
#  away from obstacles.
def getClearanceObjective(si):
    return ClearanceObjective(si)

## Create an optimization objective which attempts to optimize both
#  path length and clearance. We do this by defining our individual
#  objectives, then adding them to a MultiOptimizationObjective
#  object. This results in an optimization objective where path cost
#  is equivalent to adding up each of the individual objectives' path
#  costs.
#
#  When adding objectives, we can also optionally specify each
#  objective's weighting factor to signify how important it is in
#  optimal planning. If no weight is specified, the weight defaults to
#  1.0.
def getBalancedObjective1(si):
    lengthObj = ob.PathLengthOptimizationObjective(si)
    clearObj = ClearanceObjective(si)

    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(lengthObj, 5.0)
    opt.addObjective(clearObj, 1.0)

    return opt

## Create an optimization objective equivalent to the one returned by
#  getBalancedObjective1(), but use an alternate syntax.
#  THIS DOESN'T WORK YET. THE OPERATORS SOMEHOW AREN'T EXPORTED BY Py++.
# def getBalancedObjective2(si):
#     lengthObj = ob.PathLengthOptimizationObjective(si)
#     clearObj = ClearanceObjective(si)
#
#     return 5.0*lengthObj + clearObj


## Create an optimization objective for minimizing path length, and
#  specify a cost-to-go heuristic suitable for this optimal planning
#  problem.
def getPathLengthObjWithCostToGo(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
    return obj


# Keep these in alphabetical order and all lower case
def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        return og.BITstar(si)
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")


# Keep these in alphabetical order and all lower case
def allocateObjective(si, objectiveType):
    if objectiveType.lower() == "pathclearance":
        return getClearanceObjective(si)
    elif objectiveType.lower() == "pathlength":
        return getPathLengthObjective(si)
    elif objectiveType.lower() == "thresholdpathlength":
        return getThresholdPathLengthObj(si)
    elif objectiveType.lower() == "weightedlengthandclearancecombo":
        return getBalancedObjective1(si)
    else:
        ou.OMPL_ERROR("Optimization-objective is not implemented in allocation function.")

def plan(runTime,
         fname,
         plannerType="rrtstar",
         objectiveType="pathlength"):

    ## State space for 2DOF manipulator
    space = ob.CompoundStateSpace()
    for _ in range(ValidityChecker.number_of_joints):
        space.addSubspace(component=ob.SO2StateSpace(), weight=1)

    # Construct a space information instance for this state space
    si = ob.SpaceInformation(space)

    ## Set the object used to check which states in the space are valid
    validityChecker = ValidityChecker(si)
    si.setStateValidityChecker(validityChecker)
    si.setup()

    ## Define the start pose
    start = ob.State(space)
    start[0] = 0
    start[1] = 0
    start[2] = 0

    ## Define the goal pose
    goal = ob.State(space)
    goal[0] = np.pi/6
    goal[1] = np.pi/4
    goal[2] = np.pi/3

    print("Start joint angle:")
    print(start)

    print("Goal joint angle:")
    print(goal)

    # Create a problem instance
    pdef = ob.ProblemDefinition(si)

    # Set the start and goal states
    pdef.setStartAndGoalStates(start, goal)

    # Create the optimization objective specified by our command-line argument.
    # This helper function is simply a switch statement.
    pdef.setOptimizationObjective(allocateObjective(si, objectiveType))

    # Construct the optimal planner specified by our command line argument.
    # This helper function is simply a switch statement.
    optimizingPlanner = allocatePlanner(si, plannerType)

    # Set the problem instance for our planner to solve
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()

    # attempt to solve the planning problem in the given runtime
    solved = optimizingPlanner.solve(runTime)

    if solved:
        path_simplifier = og.PathSimplifier(si)
        path_simplifier.simplifyMax(pdef.getSolutionPath())
        
        # Output the length of the path found
        print(solved.asString())
        print('{0} found solution of path length {1:.4f} with an optimization ' \
            'objective value of {2:.4f}'.format( \
            optimizingPlanner.getName(), \
            pdef.getSolutionPath().length(), \
            pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()))

        # If a filename was specified, output the path as a matrix to
        # that file for visualization
        if fname:
            trajectory = []
            for state in pdef.getSolutionPath().getStates():
                joint_angles = [state[i].value for i in range(ValidityChecker.number_of_joints)]
                trajectory.append(joint_angles)
                
            with open(fname, 'wb') as pickle_file:
                pickle.dump(trajectory, pickle_file)

    else:
        print("No solution found.")

    return pdef.getSolutionPath()

if (__name__ == "__main__"):
    ## Create an argument parser
    parser = argparse.ArgumentParser(description='Motion planning for pick and place using PRM planning.')

    parser.add_argument('-t', '--runtime', type=float, default=1.0, help=\
        '(Optional) Specify the runtime in seconds. Defaults to 1 and must be greater than 0.')

    args = parser.parse_args()

    trajectory = plan(runTime=args.runtime,
                      fname='./rrt_trajectory.pickle')
