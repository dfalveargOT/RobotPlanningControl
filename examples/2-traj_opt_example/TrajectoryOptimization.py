import numpy as np
import scipy.sparse as sps
from scipy.linalg import block_diag
import cyipopt

class TrajectoryOptimization(cyipopt.Problem):
    ## Optimization parameters
    def __init__(self, dt, alpha=1):
        self.alpha = alpha
        self.dt = dt
        self.N = int(15/self.dt)

    def set_origin(self, origin):
        self.origin = origin

    def set_destination(self, destination):
        self.destination = destination

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles
        
    def compute_trajectory(self, x_initial):
        number_of_constraints = 2*(self.N-1) + self.N*len(self.obstacles)
        
        inf = 1.0e20
        x0 = x_initial

        lb = np.array([ -100]*2*self.N, float)
        lb[0] = self.origin[0]
        lb[self.N-1] = self.destination[0]
        lb[self.N] = self.origin[1]
        lb[2*self.N-1] = self.destination[1]
        
        ub = np.array([ 100]*2*self.N, float)
        ub[0] = self.origin[0]
        ub[self.N-1] = self.destination[0]
        ub[self.N] = self.origin[1]
        ub[2*self.N-1] = self.destination[1]

        cl = np.array([-1]*number_of_constraints, float)
        cu = np.array([1]*number_of_constraints, float)

        index = 2*(self.N-1)
        for i in range(self.N):
            for obstacle in self.obstacles:
                _, _, obstacle_radius = obstacle
            
                cl[index] = (1.1*obstacle_radius)**2
                cu[index] = inf
                index += 1

        super(TrajectoryOptimization, self).__init__(n=len(x0),
                                                     m=len(cl),
                                                     lb=lb,
                                                     ub=ub,
                                                     cl=cl,
                                                     cu=cu)

        # self.add_option('print_level', 0)
        # self.add_option('max_iter', 5000)
        
        x, info = super(TrajectoryOptimization, self).solve(x0)

        ## Convert to trajectory data
        p_x = x[:self.N]
        p_y = x[self.N:]
    
        return p_x, p_y
    
    ## ipopt required functions
    def objective(self, x):
        p_x = x[:self.N]
        p_y = x[self.N:]

        cost = self.alpha*np.sum([(p_x[n]-self.destination[0])**2 + \
                             (p_y[n]-self.destination[1])**2 for n in range(self.N)])

        cost += (1-self.alpha)*np.sum([((p_x[n]-p_x[n+1])/self.dt)**2 + \
                                  ((p_y[n]-p_y[n+1])/self.dt)**2 for n in range(self.N-1)])
        
        return cost

    def gradient(self, x):
        p_x = x[:self.N]
        p_y = x[self.N:]

        gradient_cost = self.alpha*np.array([2*(p_x[n]-self.destination[0]) for n in range(self.N)] + \
                                       [2*(p_y[n]-self.destination[1]) for n in range(self.N)])

        gradient_cost += (1-self.alpha)*np.array([2*(p_x[n]-p_x[n+1])/(self.dt**2) for n in range(self.N-1)] + [0] + \
                                            [2*(p_y[n]-p_y[n+1])/(self.dt**2) for n in range(self.N-1)] + [0])

        gradient_cost += (1-self.alpha)*np.array([0] + [-2*(p_x[n]-p_x[n+1])/(self.dt**2) for n in range(self.N-1)] + \
                                            [0] + [-2*(p_y[n]-p_y[n+1])/(self.dt**2) for n in range(self.N-1)])
        
        return gradient_cost

    def constraints(self, x):
        p_x = x[:self.N]
        p_y = x[self.N:]
        
        constraints = []
        
        ## Velocity constraints
        for n in range(self.N-1):
            constraints.append((p_x[n] - p_x[n+1])/self.dt) # x-axis

        for n in range(self.N-1):
            constraints.append((p_y[n] - p_y[n+1])/self.dt) # y-axis

        ## Obstacle avoidance constraints
        for n in range(self.N):
            for obstacle in self.obstacles:
                obstacle_x, obstacle_y, obstacle_radius = obstacle
                distance_to_obstacle = (p_x[n] - obstacle_x)**2 + (p_y[n] - obstacle_y)**2
        
                constraints.append(distance_to_obstacle)

        return np.array(constraints)

    def jacobianstructure(self):
        number_of_constraints = 2*(self.N-1) + self.N*len(self.obstacles)
        jacobian_structure = np.zeros((number_of_constraints, 2*self.N), int)

        ## Velocity constraint part
        index = 0
        for n in range(self.N-1):
            jacobian_structure[index, n] = 1
            jacobian_structure[index, n+1] = 1
            index += 1
            
        for n in range(self.N-1):
            jacobian_structure[index, self.N+n] = 1
            jacobian_structure[index, self.N+n+1] = 1
            index += 1

        ## Obstacle avodiance part
        for n in range(self.N):
            for obstacle in self.obstacles:
                jacobian_structure[index, n] = 1
                jacobian_structure[index, self.N+n] = 1

                index += 1

        return np.nonzero(jacobian_structure)

    def jacobian(self, x):
        p_x = x[:self.N]
        p_y = x[self.N:]
        
        number_of_constraints = 2*(self.N-1) + self.N*len(self.obstacles)
        jacobian_matrix = np.zeros((number_of_constraints, 2*self.N), float)

        ## Velocity constraint part
        index = 0
        for n in range(self.N-1):
            jacobian_matrix[index, n] = 1/self.dt
            jacobian_matrix[index, n+1] = -1/self.dt
            index += 1
            
        for n in range(self.N-1):
            jacobian_matrix[index, self.N+n] = 1/self.dt
            jacobian_matrix[index, self.N+n+1] = -1/self.dt
            index += 1

        ## Obstacle avodiance part
        for n in range(self.N):
            for obstacle in self.obstacles:
                obstacle_x, obstacle_y, obstacle_radius = obstacle
                
                jacobian_matrix[index, n] = 2*(p_x[n] - obstacle_x)
                jacobian_matrix[index, self.N+n] = 2*(p_y[n] - obstacle_y)

                index += 1

        row, col = self.jacobianstructure()
        
        return jacobian_matrix[row, col]

    def hessianstructure(self):
        hessian_structure = sps.coo_matrix(np.tril(np.ones((2*self.N, 2*self.N))))

        return (hessian_structure.row, hessian_structure.col)

    def hessian(self, x, lagrange, obj_factor):
        hessian_cost = self.alpha*2*np.eye(2*self.N)

        hessian_cost[0,0] += (1-self.alpha)/(self.dt**2)*2
        hessian_cost[0,1] += -(1-self.alpha)/(self.dt**2)*2
        for n in range(1, self.N-1):
            hessian_cost[n,n-1] += -(1-self.alpha)/(self.dt**2)*2
            hessian_cost[n,n] += (1-self.alpha)/(self.dt**2)*4
            hessian_cost[n,n+1] += -(1-self.alpha)/(self.dt**2)*2

        hessian_cost[self.N-1,self.N-2] += (1-self.alpha)/(self.dt**2)*2
        hessian_cost[self.N-1,self.N-1] += -(1-self.alpha)/(self.dt**2)*2

        hessian_cost[self.N,self.N] += (1-self.alpha)/(self.dt**2)*2
        hessian_cost[self.N,self.N+1] += -(1-self.alpha)/(self.dt**2)*2
        for n in range(self.N-1, 2*self.N-1):
            hessian_cost[n,n-1] += -(1-self.alpha)/(self.dt**2)*2
            hessian_cost[n,n] += (1-self.alpha)/(self.dt**2)*4
            hessian_cost[n,n+1] += -(1-self.alpha)/(self.dt**2)*2

        hessian_cost[2*self.N-1,2*self.N-2] += (1-self.alpha)/(self.dt**2)*2
        hessian_cost[2*self.N-1,2*self.N-1] += -(1-self.alpha)/(self.dt**2)*2
            
        hessian = obj_factor*hessian_cost

        ## Obstacle avodiance part
        index = 2*(self.N-1)
        for n in range(self.N):
            for obstacle in self.obstacles:
                hessian_i = np.zeros((2*self.N, 2*self.N))
                hessian_i[n, n] = 2
                hessian_i[self.N+n, self.N+n] = 2

                hessian += lagrange[index]*hessian_i

                index += 1

        row, col = self.hessianstructure()
        
        return hessian[row, col]
