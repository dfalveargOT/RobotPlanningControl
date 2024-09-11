import numpy as np
import scipy.sparse as sps
from scipy.linalg import block_diag
import cyipopt

class MpcRacecarRobot(cyipopt.Problem):
    car_length = 0.325
    
    N_state = 4
    N_input = 2
    T_horizon = 10*3
    h = 1/10

    def __init__(self):
        self.number_of_variables = (self.N_state + self.N_input)*self.T_horizon
        self.number_of_constraints = self.N_state*(self.T_horizon-1)

        self.weight_x = 2.0
        self.weight_y = 2.0
        self.weight_ctheta = 1
        self.weight_stheta = 1
        self.weight_v = .1
        self.weight_u = .15

    def set_cost_weights(self, weights):
        self.weight_x, self.weight_y, self.weight_ctheta, self.weight_stheta, self.weight_v, self.weight_u = weights

    def set_reference_trajectory(self, reference_trajectory):
        self.reference_trajectory = reference_trajectory
        self.reference_trajectory_x, self.reference_trajectory_y, self.reference_trajectory_theta = zip(*self.reference_trajectory)
        self.reference_trajectory_ctheta = np.cos(self.reference_trajectory_theta)
        self.reference_trajectory_stheta = np.sin(self.reference_trajectory_theta)
        
    def initialize_mpc(self, initial_time_instance, initial_pose):
        self.initial_time_instance = initial_time_instance
        self.initial_pose = [initial_pose[0], initial_pose[1], np.cos(initial_pose[2]), np.sin(initial_pose[2])]

    def compute_mpc(self, x_initial):
        inf     = 1.0e20
        x0 = x_initial

        lb = np.array([ -inf]*self.number_of_variables,float)
        ub = np.array([ inf]*self.number_of_variables,float)

        lb[0] = self.initial_pose[0]
        lb[self.T_horizon] = self.initial_pose[1]
        lb[2*self.T_horizon] = self.initial_pose[2]
        lb[3*self.T_horizon] = self.initial_pose[3]
        ub[0] = self.initial_pose[0]
        ub[self.T_horizon] = self.initial_pose[1]
        ub[2*self.T_horizon] = self.initial_pose[2]
        ub[3*self.T_horizon] = self.initial_pose[3]

        for t in range(self.T_horizon):
            lb[4*self.T_horizon+t] = 0
            ub[4*self.T_horizon+t] = 1
    
            lb[5*self.T_horizon+t] = -5
            ub[5*self.T_horizon+t] = 5

        cl = np.array([0]*self.number_of_constraints, float)
        cu = np.array([0]*self.number_of_constraints, float)

        super(MpcRacecarRobot, self).__init__(n=len(x0),
                                         m=len(cl),
                                         lb=lb,
                                         ub=ub,
                                         cl=cl,
                                         cu=cu
                                         )

        self.add_option('print_level', 0)
        self.add_option('max_iter', 4)
        
        x, info = super(MpcRacecarRobot, self).solve(x_initial)

        p_x = x[0*self.T_horizon:1*self.T_horizon]
        p_y = x[1*self.T_horizon:2*self.T_horizon]
        ctheta = x[2*self.T_horizon:3*self.T_horizon]
        stheta = x[3*self.T_horizon:4*self.T_horizon]
        theta = np.arctan2(stheta, ctheta)
        v = x[4*self.T_horizon:5*self.T_horizon]
        u = x[5*self.T_horizon:6*self.T_horizon]

        return p_x, p_y, theta, v, u

    ## ipopt required functions
    def objective(self, x):
        p_x = x[0*self.T_horizon:1*self.T_horizon]
        p_y = x[1*self.T_horizon:2*self.T_horizon]
        ctheta = x[2*self.T_horizon:3*self.T_horizon]
        stheta = x[3*self.T_horizon:4*self.T_horizon]
        v = x[4*self.T_horizon:5*self.T_horizon]
        u = x[5*self.T_horizon:6*self.T_horizon]

        reference_trajectory_x = self.reference_trajectory_x[self.initial_time_instance:self.initial_time_instance+self.T_horizon]
        reference_trajectory_y = self.reference_trajectory_y[self.initial_time_instance:self.initial_time_instance+self.T_horizon]
        reference_trajectory_ctheta = self.reference_trajectory_ctheta[self.initial_time_instance:self.initial_time_instance+self.T_horizon]
        reference_trajectory_stheta = self.reference_trajectory_stheta[self.initial_time_instance:self.initial_time_instance+self.T_horizon]
        weight_x = self.weight_x
        weight_y = self.weight_y
        weight_ctheta = self.weight_ctheta
        weight_stheta = self.weight_stheta
        weight_v = self.weight_v
        weight_u = self.weight_u

        ##### EXERCISE #####
        ## Compute the cost function over the time interval [0:self.T_horizon] using
        ## 1. the reference trajectory:
        ##    reference_trajectory_x[t: t+self.T_horizon],
        ##    reference_trajectory_y[t: t+self.T_horizon],
        ##    reference_trajectory_ctheta[t: t+self.T_horizon]
        ##    reference_trajectory_stheta[t: t+self.T_horizon]
        ## 2. weights: weight_x, weight_y, weight_ctheta, weight_stheta, weight_v, weight_u
        cost = np.sum([weight_x*(p_x[t] - reference_trajectory_x[t])**2 + \
            weight_y*(p_y[t] - reference_trajectory_y[t])**2 + weight_ctheta*(ctheta[t] - reference_trajectory_ctheta[t])**2 + \
                weight_stheta*(stheta[t] - reference_trajectory_stheta[t])**2 + weight_v*(v**2) + weight_u*(u**2) for t in range(self.T_horizon)])

        ##### /EXERCISE #####

        cost += np.sum([10*(ctheta[t]**2+stheta[t]**2-1)**2 for t in range(self.T_horizon)])

        return cost

    def gradient(self, x):
        p_x = x[0*self.T_horizon:1*self.T_horizon]
        p_y = x[1*self.T_horizon:2*self.T_horizon]
        ctheta = x[2*self.T_horizon:3*self.T_horizon]
        stheta = x[3*self.T_horizon:4*self.T_horizon]
        v = x[4*self.T_horizon:5*self.T_horizon]
        u = x[5*self.T_horizon:6*self.T_horizon]

        reference_trajectory_x = self.reference_trajectory_x[self.initial_time_instance:self.initial_time_instance+self.T_horizon]
        reference_trajectory_y = self.reference_trajectory_y[self.initial_time_instance:self.initial_time_instance+self.T_horizon]
        reference_trajectory_ctheta = self.reference_trajectory_ctheta[self.initial_time_instance:self.initial_time_instance+self.T_horizon]
        reference_trajectory_stheta = self.reference_trajectory_stheta[self.initial_time_instance:self.initial_time_instance+self.T_horizon]
        weight_x = self.weight_x
        weight_y = self.weight_y
        weight_ctheta = self.weight_ctheta
        weight_stheta = self.weight_stheta
        weight_v = self.weight_v
        weight_u = self.weight_u

        ##### EXERCISE #####
        ## Compute the gradient of the cost function over the time interval [0:self.T_horizon] using
        ## 1. the reference trajectory:
        ##    reference_trajectory_x[t: t+self.T_horizon],
        ##    reference_trajectory_y[t: t+self.T_horizon],
        ##    reference_trajectory_ctheta[t: t+self.T_horizon]
        ##    reference_trajectory_stheta[t: t+self.T_horizon]
        ## 2. weights: weight_x, weight_y, weight_ctheta, weight_stheta, weight_v, weight_u

        gradient_cost_x = [2*weight_x*(p_x[t] - reference_trajectory_x[t]) for t in range(self.T_horizon)]
        gradient_cost_y = [2*weight_y*(p_y[t] - reference_trajectory_y[t]) for t in range(self.T_horizon)]
        gradient_cost_ctheta = [2*weight_ctheta*(ctheta[t] - reference_trajectory_ctheta[t]) for t in range(self.T_horizon)]
        gradient_cost_stheta = [2*weight_stheta*(stheta[t] - reference_trajectory_stheta[t]) for t in range(self.T_horizon)]
        gradient_cost_v = [2*weight_v*v[t] for t in range(self.T_horizon)]
        gradient_cost_u = [2*weight_u*u[t] for t in range(self.T_horizon)]

        ##### /EXERCISE #####

        gradient_cost_ctheta += np.array([40*ctheta[t]*(ctheta[t]**2+stheta[t]**2-1) for t in range(self.T_horizon)])
        gradient_cost_stheta += np.array([40*stheta[t]*(ctheta[t]**2+stheta[t]**2-1) for t in range(self.T_horizon)])
        gradient_cost = np.array(list(gradient_cost_x) + \
                                 list(gradient_cost_y) + \
                                 list(gradient_cost_ctheta) + \
                                 list(gradient_cost_stheta) + \
                                 list(gradient_cost_v) + \
                                 list(gradient_cost_u))

        return gradient_cost

    def constraints(self, x):
        p_x = x[0*self.T_horizon:1*self.T_horizon]
        p_y = x[1*self.T_horizon:2*self.T_horizon]
        ctheta = x[2*self.T_horizon:3*self.T_horizon]
        stheta = x[3*self.T_horizon:4*self.T_horizon]
        v = x[4*self.T_horizon:5*self.T_horizon]
        u = x[5*self.T_horizon:6*self.T_horizon]

        h = self.h
        L = self.car_length
        
        constraints = np.array([])
        for t in range(self.T_horizon-1):
            ##### EXERCISE #####
            ## Compute the constraint (derived from the kinematics)
            ## h: time discretization interval
            w_t = (v[t]*np.tan(u[t]))/L
            g_x_t = p_x[t+1] - p_x[t] - h*v[t]*ctheta[t]
            g_y_t = p_y[t+1] - p_y[t] - h*v[t]*stheta[t]
            g_ctheta_t = ctheta[t+1] - ctheta[t] + h*w_t*stheta[t]
            g_stheta_t = stheta[t+1] - stheta[t] - h*w_t*ctheta[t]
            
            ##### /EXERCISE #####

            kinematics = np.array([g_x_t, g_y_t, g_ctheta_t, g_stheta_t])
            constraints = np.concatenate((constraints, kinematics))

        return constraints

    def jacobianstructure(self):
        jacobian_structure = np.zeros((self.number_of_constraints, self.number_of_variables), int)

        for t in range(self.T_horizon-1):
            ## p_x dynamics
            jacobian_structure[t*self.N_state, t+1] = 1 # p_x[t+1]
            jacobian_structure[t*self.N_state, t] = 1 # p_x[t]
            jacobian_structure[t*self.N_state, 2*self.T_horizon+t] = 1 # ctheta[t]
            jacobian_structure[t*self.N_state, 4*self.T_horizon+t] = 1 # v[t]
            
            ## p_y dynamics
            jacobian_structure[t*self.N_state+1, self.T_horizon+t+1] = 1 # p_y[t+1]
            jacobian_structure[t*self.N_state+1, self.T_horizon+t] = 1 # p_y[t]
            jacobian_structure[t*self.N_state+1, 3*self.T_horizon+t] = 1 # stheta[t]
            jacobian_structure[t*self.N_state+1, 4*self.T_horizon+t] = 1 # v[t]
            
            ## ctheta dynamics
            jacobian_structure[t*self.N_state+2, 2*self.T_horizon+t+1] = 1 # ctheta[t+1]
            jacobian_structure[t*self.N_state+2, 2*self.T_horizon+t] = 1 # ctheta[t]
            jacobian_structure[t*self.N_state+2, 3*self.T_horizon+t] = 1 # stheta[t]
            jacobian_structure[t*self.N_state+2, 4*self.T_horizon+t] = 1 # v[t]
            jacobian_structure[t*self.N_state+2, 5*self.T_horizon+t] = 1 # u[t]

            ## stheta dynamics
            jacobian_structure[t*self.N_state+3, 3*self.T_horizon+t+1] = 1 # stheta[t+1]
            jacobian_structure[t*self.N_state+3, 3*self.T_horizon+t] = 1 # stheta[t]
            jacobian_structure[t*self.N_state+3, 2*self.T_horizon+t] = 1 # ctheta[t]
            jacobian_structure[t*self.N_state+3, 4*self.T_horizon+t] = 1 # v[t]
            jacobian_structure[t*self.N_state+3, 5*self.T_horizon+t] = 1 # u[t]

        return np.nonzero(jacobian_structure)

    def jacobian(self, x):
        p_x = x[0*self.T_horizon:1*self.T_horizon]
        p_y = x[1*self.T_horizon:2*self.T_horizon]
        ctheta = x[2*self.T_horizon:3*self.T_horizon]
        stheta = x[3*self.T_horizon:4*self.T_horizon]
        v = x[4*self.T_horizon:5*self.T_horizon]
        u = x[5*self.T_horizon:6*self.T_horizon]

        h = self.h
        L = self.car_length
        
        jacobian = np.zeros((self.number_of_constraints, self.number_of_variables), float)
        for t in range(self.T_horizon-1):
            ##### EXERCISE #####
            ## Compute the jacobian of g_x_t with respect to (p_x[t+1], p_x[t], ctheta[t], v[t])
            jacobian_g_x_t = [1,
                              -1,
                              -h*v[t],
                              -h*ctheta[t]]

            ## Compute the jacobian of g_y_t with respect to (p_y[t+1], p_y[t], stheta[t], v[t])
            jacobian_g_y_t = [1,
                              -1,
                              -h*v[t],
                              -h*stheta[t]]

            ## Compute the jacobian of g_x_t with respect to (ctheta[t+1], ctheta[t], stheta[t], v[t], u[t])
            jacobian_g_ctheta_t = [1,
                                   -1,
                                   (h*v[t]*np.tan(u[t]))*(1/L),
                                   (h*np.tan(u[t])*stheta[t])*(1/L),
                                   (h*v[t]*(1/np.cos(u[t])**2)*stheta[t])*(1/L)]

            ## Compute the jacobian of g_x_t with respect to (stheta[t+1], stheta[t], ctheta[t], v[t], u[t])
            jacobian_g_stheta_t = [1,
                                   -1,
                                   -(h*v[t]*np.tan(u[t]))*(1/L),
                                   -(h*np.tan(u[t])*ctheta[t])*(1/L),
                                   -(h*v[t]*(1/np.cos(u[t])**2)*ctheta[t])*(1/L)]

            ##### /EXERCISE #####

            ## p_x dynamics
            jacobian[t*self.N_state, t+1] = jacobian_g_x_t[0] # p_x[t+1]
            jacobian[t*self.N_state, t] = jacobian_g_x_t[1] # p_x[t]
            jacobian[t*self.N_state, 2*self.T_horizon+t] = jacobian_g_x_t[2] # ctheta[t]
            jacobian[t*self.N_state, 4*self.T_horizon+t] = jacobian_g_x_t[3] # v[t]
                        
            ## p_y dynamics
            jacobian[t*self.N_state+1, self.T_horizon+t+1] = jacobian_g_y_t[0] # p_y[t+1]
            jacobian[t*self.N_state+1, self.T_horizon+t] = jacobian_g_y_t[1] # p_y[t]
            jacobian[t*self.N_state+1, 3*self.T_horizon+t] = jacobian_g_y_t[2] # stheta[t]
            jacobian[t*self.N_state+1, 4*self.T_horizon+t] = jacobian_g_y_t[3] # v[t]
            
            ## ctheta dynamics
            jacobian[t*self.N_state+2, 2*self.T_horizon+t+1] = jacobian_g_ctheta_t[0] # ctheta[t+1]
            jacobian[t*self.N_state+2, 2*self.T_horizon+t] = jacobian_g_ctheta_t[1] # ctheta[t]
            jacobian[t*self.N_state+2, 3*self.T_horizon+t] = jacobian_g_ctheta_t[2] # stheta[t]
            jacobian[t*self.N_state+2, 4*self.T_horizon+t] = jacobian_g_ctheta_t[3] # v[t]
            jacobian[t*self.N_state+2, 5*self.T_horizon+t] = jacobian_g_ctheta_t[4] # u[t]

            ## stheta dynamics
            jacobian[t*self.N_state+3, 3*self.T_horizon+t+1] = jacobian_g_stheta_t[0] # stheta[t+1]
            jacobian[t*self.N_state+3, 3*self.T_horizon+t] = jacobian_g_stheta_t[1] # ctheta[t]
            jacobian[t*self.N_state+3, 2*self.T_horizon+t] = jacobian_g_stheta_t[2] # ctheta[t]
            jacobian[t*self.N_state+3, 4*self.T_horizon+t] = jacobian_g_stheta_t[3] # v[t]
            jacobian[t*self.N_state+3, 5*self.T_horizon+t] = jacobian_g_stheta_t[4] # u[t]

        row, col = self.jacobianstructure()
        
        return jacobian[row, col]

    def hessianstructure(self):
        hessian_structure = sps.coo_matrix(np.tril(np.ones((self.number_of_variables, self.number_of_variables))))

        return (hessian_structure.row, hessian_structure.col)
        

    def hessian(self, x, lagrange, obj_factor):
        p_x = x[0*self.T_horizon:1*self.T_horizon]
        p_y = x[1*self.T_horizon:2*self.T_horizon]
        ctheta = x[2*self.T_horizon:3*self.T_horizon]
        stheta = x[3*self.T_horizon:4*self.T_horizon]
        v = x[4*self.T_horizon:5*self.T_horizon]
        u = x[5*self.T_horizon:6*self.T_horizon]

        weight_x = self.weight_x
        weight_y = self.weight_y
        weight_ctheta = self.weight_ctheta
        weight_stheta = self.weight_stheta
        weight_v = self.weight_v
        weight_u = self.weight_u

        h = self.h
        L = self.car_length

        ##### EXERCISE #####
        ## Compute the hessian for cost function
        hessian_cost_xx = 2*weight_x
        hessian_cost_yy = 2*weight_y
        hessian_cost_ctct = 2*weight_ctheta 
        hessian_cost_stst = 2*weight_stheta
        hessian_cost_vv = 2*weight_v
        hessian_cost_uu = 2*weight_u

        ##### /EXERCISE #####

        hessian_cost = block_diag(hessian_cost_xx*np.eye(self.T_horizon),
                                  hessian_cost_yy*np.eye(self.T_horizon),
                                  hessian_cost_ctct*np.eye(self.T_horizon),
                                  hessian_cost_stst*np.eye(self.T_horizon),
                                  hessian_cost_vv*np.eye(self.T_horizon),
                                  hessian_cost_uu*np.eye(self.T_horizon))
        
        hessian = obj_factor*hessian_cost

        for t in range(self.T_horizon-1):
            ##### EXERCISE #####
            ## Compute the Hessian for the constraint g_x_t with respect to (ctheta[t], v[t])
            hessian_g_x_t_ctv = -h

            ## Compute the Hessian for the constraint g_y_t with respect to (stheta[t], v[t])
            hessian_g_y_t_stv = -h

            ## Compute the Hessian for the constraint g_ctheta_t with respect to (stheta[t], v[t])
            hessian_g_ctheta_t_stv = h*np.tan(u[t])*(1/L)

            ## Compute the Hessian for the constraint g_ctheta_t with respect to (stheta[t], u[t])
            hessian_g_ctheta_t_stu = h*v[t]*(1/np.cos(u[t])**2)*(1/L)

            ## Compute the Hessian for the constraint g_ctheta_t with respect to (v[t], u[t])
            hessian_g_ctheta_t_vu = h*(1/np.cos(u[t])**2)*stheta[t]*(1/L)

            ## Compute the Hessian for the constraint g_ctheta_t with respect to (u[t], u[t])
            hessian_g_ctheta_t_uu = 2*h*v[t]*(1/np.cos(u[t])**2)*np.tan(u[t])*stheta[t]*(1/L)

            ## Compute the Hessian for the constraint g_stheta_t with respect to (ctheta[t], v[t])
            hessian_g_stheta_t_ctv = -h*np.tan(u[t])*(1/L)

            ## Compute the Hessian for the constraint g_stheta_t with respect to (ctheta[t], u[t])
            hessian_g_stheta_t_ctu = -h*v[t]*(1/np.cos(u[t])**2)*(1/L)

            ## Compute the Hessian for the constraint g_stheta_t with respect to (v[t], u[t])
            hessian_g_stheta_t_vu = -h*(1/np.cos(u[t])**2)*ctheta[t]*(1/L)

            ## Compute the Hessian for the constraint g_stheta_t with respect to (u[t], u[t])
            hessian_g_stheta_t_uu = -2*h*v[t]*(1/np.cos(u[t])**2)*np.tan(u[t])*ctheta[t]*(1/L)

            ##### /EXERCISE #####
            
            ## Hessian for p_x
            hessian_1 = np.zeros((self.number_of_variables, self.number_of_variables))
            hessian_1[2*self.T_horizon+t, 4*self.T_horizon+t] = hessian_g_x_t_ctv
            hessian_1[4*self.T_horizon+t, 2*self.T_horizon+t] = hessian_g_x_t_ctv
            
            hessian += lagrange[t*self.N_state]*hessian_1

            ## Hessian for p_y
            hessian_2 = np.zeros((self.number_of_variables, self.number_of_variables))
            hessian_2[3*self.T_horizon+t, 4*self.T_horizon+t] = hessian_g_y_t_stv
            hessian_2[4*self.T_horizon+t, 3*self.T_horizon+t] = hessian_g_y_t_stv

            hessian += lagrange[t*self.N_state+1]*hessian_2

            ## Hessian for ctheta
            hessian_3 = np.zeros((self.number_of_variables, self.number_of_variables))
            hessian_3[3*self.T_horizon+t, 4*self.T_horizon+t] = hessian_g_ctheta_t_stv
            hessian_3[4*self.T_horizon+t, 3*self.T_horizon+t] = hessian_g_ctheta_t_stv
            hessian_3[3*self.T_horizon+t, 5*self.T_horizon+t] = hessian_g_ctheta_t_stu
            hessian_3[5*self.T_horizon+t, 3*self.T_horizon+t] = hessian_g_ctheta_t_stu
            hessian_3[4*self.T_horizon+t, 5*self.T_horizon+t] = hessian_g_ctheta_t_vu
            hessian_3[5*self.T_horizon+t, 4*self.T_horizon+t] = hessian_g_ctheta_t_vu
            hessian_3[5*self.T_horizon+t, 5*self.T_horizon+t] = hessian_g_ctheta_t_uu
            
            hessian += lagrange[t*self.N_state+2]*hessian_3

            ## Hessian for stheta
            hessian_4 = np.zeros((self.number_of_variables, self.number_of_variables))
            hessian_4[2*self.T_horizon+t, 4*self.T_horizon+t] = hessian_g_stheta_t_ctv
            hessian_4[4*self.T_horizon+t, 2*self.T_horizon+t] = hessian_g_stheta_t_ctv
            hessian_4[2*self.T_horizon+t, 5*self.T_horizon+t] = hessian_g_stheta_t_ctu
            hessian_4[5*self.T_horizon+t, 2*self.T_horizon+t] = hessian_g_stheta_t_ctu
            hessian_4[4*self.T_horizon+t, 5*self.T_horizon+t] = hessian_g_stheta_t_vu
            hessian_4[5*self.T_horizon+t, 4*self.T_horizon+t] = hessian_g_stheta_t_vu
            hessian_4[5*self.T_horizon+t, 5*self.T_horizon+t] = hessian_g_stheta_t_uu
            
            hessian += lagrange[t*self.N_state+3]*hessian_4

        row, col = self.hessianstructure()
        return hessian[row, col]

    ##########
