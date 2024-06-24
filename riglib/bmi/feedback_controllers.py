'''
Feedback controllers. In the BMI context, these have three potential
applications: (1) assisitve/shared control between neural and pure machine sources,
(2) estimating the "intended" BMI movements of the subject (see clda.py), and
(3) driving populations of simulated neurons with artificially constructed tuning
functions. 
'''
import numpy as np


class FeedbackController(object):
    '''
    Abstract class for feedback controllers, only used for type-checking & interface standardization
    '''
    def __init__(self, *args, **kwargs):
        pass

    def calc_next_state(self, current_state, target_state, mode=None):
        raise NotImplementedError

    def __call__(self, current_state, target_state, mode=None):
        raise NotImplementedError

    def get(self, current_state, target_state, mode=None):
        raise NotImplementedError


class LinearFeedbackController(FeedbackController):
    '''
    Generic linear state-feedback controller. Can be time-varying in general and not be related to a specific cost function
    '''
    def __init__(self, A, B, F):
        '''
        Constructor for LinearFeedbackController
        
        FC for a linear system
            x_{t+1} = Ax_t + Bu_t
        where the control input u_t is calculated using linear feedback
            u_t = F(x_t - x^*)

        Parameters
        ----------
        B : np.mat
            Input matrix for the system
        F : np.mat
            static feedback gain matrix

        Returns
        -------
        LinearFeedbackController instance
        '''
        self.A = A
        self.B = B
        self.F = F 

    def calc_next_state(self, current_state, target_state, mode=None):
        '''
        Returns x_{t+1} = Ax_t + BF(x* - x_t)

        Parameters
        ----------
        current_state : np.matrix
            Current state of the system, x_t 
        target_state : np.matrix
            State that you're trying to steer the system toward, x*
        mode : object, default=None
            Select the operational mode of the feedback controller. Ignored in this class

        Returns
        -------
        np.matrix
        '''
        # explicitly cast current_state and target_state to column vectors
        current_state = np.mat(current_state).reshape(-1,1)
        target_state = np.mat(target_state).reshape(-1,1)
        ns = self.A * current_state + self.B * self.F * (target_state - current_state)

        return ns

    def __call__(self, current_state, target_state, mode=None):
        '''
        Calculate the control input u_t = BF(x* - x_t)

        Parameters
        ----------
        (see self.calc_next_state for input argument descriptions)

        Returns
        -------
        np.mat of shape (N, 1)
            B*u where u_t = F(x^* - x_t)
        '''
        
        
        current_state = np.mat(current_state).reshape(-1,1)
        target_state = np.mat(target_state).reshape(-1,1)
        Bu = self.B * self.F * (target_state - current_state)
        return Bu


class LQRController(LinearFeedbackController):
    '''Linear feedback controller where control gains are set by optimizing a quadratic cost function'''
    def __init__(self, A, B, Q, R, **kwargs):
        '''
        Constructor for LQRController

        The system should evolve as
        $$x_{t+1} = Ax_t + Bu_t + w_t; w_t ~ N(0, W)$$

        with infinite horizon cost 
        $$\sum{t=0}^{+\infty} (x_t - x_target)^T * Q * (x_t - x_target) + u_t^T * R * u_t$$

        Parameters
        ----------
        A: np.ndarray of shape (n_states, n_states)
            Model of the state transition matrix of the system to be controlled. 
        B: np.ndarray of shape (n_states, n_controls)
            Control input matrix of the system to be controlled. 
        Q: np.ndarray of shape (n_states, n_states)
            Quadratic cost on state
        R: np.ndarray of shape (n_controls, n_controls)
            Quadratic cost on control inputs

        Returns
        -------
        LQRController instance
        '''
        self.A = np.mat(A)
        self.B = np.mat(B)
        self.Q = np.mat(Q)
        self.R = np.mat(R)
        F = self.dlqr(A, B, Q, R, **kwargs)
        super(LQRController, self).__init__(A, B, F, **kwargs)

    @staticmethod
    def dlqr(A, B, Q, R, Q_f=None, T=np.inf, max_iter=1000, eps=1e-10, dtype=np.mat):
        '''
        Find the solution to the discrete-time LQR problem

        The system should evolve as
        $$x_{t+1} = Ax_t + Bu_t + w_t; w_t ~ N(0, W)$$

        with cost function
        $$\sum{t=0}^{T} (x_t - x_target)^T * Q * (x_t - x_target) + u_t^T * R * u_t$$

        The cost function can be either finite or infinite horizion, where finite horizion is assumed if 
        a final const is specified

        Parameters
        ----------
        A: np.ndarray of shape (n_states, n_states)
            Model of the state transition matrix of the system to be controlled. 
        B: np.ndarray of shape (n_states, n_controls)
            Control input matrix of the system to be controlled. 
        Q: np.ndarray of shape (n_states, n_states)
            Quadratic cost on state
        R: np.ndarray of shape (n_controls, n_controls)
            Quadratic cost on control inputs
        Q_f: np.ndarray of shape (n_states, n_states), optional, default=None
            Final quadratic cost on state at the end of the horizon. Only applies to finite-horizion variants
        T: int, optional, default = np.inf
            Control horizon duration. Infinite by default. Must be less than infinity (and Q_f must be specified)
            to get the finite horizon feedback controllers
        eps: float, optional, default=1e-10
            Threshold of change in feedback matrices to define when the Riccatti recursion has converged
        dtype: callable, optional, default=np.mat
            Callable function to reformat the feedback matrices 

        Returns
        -------
        K: list or matrix
            Returns a sequence of feedback gains if finite horizon or a single controller if infinite horizon.

        '''
        if Q_f == None: 
            Q_f = Q

        if T < np.inf: # Finite horizon
            K = [None]*T
            P = Q_f
            for t in range(0,T-1)[::-1]:
                K[t] = (R + B.T*P*B).I * B.T*P*A
                P = Q + A.T*P*A -A.T*P*B*K[t]
            return dtype(K)
        else: # Infinite horizon
            P = Q_f
            K = np.inf
            for t in range(max_iter):
                K_old = K
                K = (R + B.T*P*B).I * B.T*P*A
                P = Q + A.T*P*A -A.T*P*B*K 
                if np.linalg.norm(K - K_old) < eps:
                    break
            return dtype(K)


class MultiModalLFC(LinearFeedbackController):
    '''
    A linear feedback controller with different feedback gains in different "modes"
    '''
    def __init__(self, A=None, B=None, F_dict=dict()):
        '''
        Constructor for MultiModalLFC

        Parameters
        ----------
        B : np.mat
            Input matrix for the system
        F : dict
            keys should be control 'modes', values should be np.mat static feedback gain matrices

        Returns
        -------
        MultiModalLFC instance
        '''
        self.A = A
        self.B = B
        self.F_dict = F_dict
        self.F = None

    def calc_next_state(self, current_state, target_state, mode=None):
        '''
        See LinearFeedbackController.calc_next_state for docs
        '''
        self.F = self.F_dict[mode]
        super(MultiModalLFC, self).calc_next_state(current_state, target_state, mode=mode)

    def __call__(self, current_state, target_state, mode=None):
        '''
        See LinearFeedbackController.__call__ for docs
        '''        
        self.F = self.F_dict[mode]
        super(MultiModalLFC, self).__call__(current_state, target_state, mode=mode)        


class PIDController(FeedbackController):
    '''
    Linear feedback controller where gains are set directly instead of through a cost function
    '''
    def __init__(self, K_prop, K_deriv, K_int, state_order):
        '''
        Constructor for PIDController

        Parameters
        ----------
        K_prop : float 
            Gain on proportional error
        K_deriv : float 
            Gain on derivative error 
        K_int : float
            Gain on integrated error
        state_order : np.ndarray of shape (N, 1)
            Specify whether each element of the state vector is a proportional, derivative, or integral state

        Returns
        -------
        PIDController instance
        '''
        self.K_prop = K_prop
        self.K_deriv = K_deriv
        self.K_int = K_int

        self.int_terms, = np.nonzero(state_order == -1)
        self.prop_terms, = np.nonzero(state_order == 0)
        self.deriv_terms, = np.nonzero(state_order == 1)

    def __call__(self, current_state, target_state):
        '''
        Determine the PID controller output to be added onto the current state

        Parameters
        ----------
        current_state : np.matrix
            Current state of the system, x_t 
        target_state : np.matrix
            State that you're trying to steer the system toward, x*

        Returns
        -------
        cmd : np.ndarray of shape (K, 1)
            K is the number of states in the proportional term. 
        '''        
        state_diff = target_state - current_state
        cmd = 0
        if len(self.prop_terms) > 0:
            cmd += self.K_prop * state_diff[self.prop_terms] 

        if len(self.deriv_terms) > 0:
            cmd += self.K_deriv * state_diff[self.deriv_terms] 

        if len(self.int_terms) > 0:
            cmd += self.K_int * state_diff[self.int_terms]

        return cmd

    def calc_next_state(self, current_state, target_state, **kwargs):
        '''
        see self.__call__
        '''
        return self.__call__(current_state, target_state)


######################################################################
##### Deprecated task/plant-specific controllers for simulations #####
######################################################################
class CenterOutCursorGoal(object):
    '''
    Cursor controller which moves the cursor toward the target at a constant speed
    '''
    def __init__(self, angular_noise_var=0, gain=0.15):
        '''
        Constructor for CenterOutCursorGoal

        Parameters
        ----------
        angular_noise_var: float, optional, default=0
            Angular noise is added onto the control direction as a clipped Gaussian distribution with this variance
        gain: float, optional, default=0.15
            Speed at which to move the cursor, in m/s

        Returns
        -------
        CenterOutCursorGoal instance
        '''
        self.angular_noise_var = angular_noise_var
        self.gain = gain

    def get(self, cur_target, cur_pos, keys_pressed=None):
        # Make sure y-dimension is 0
        assert cur_pos[1] == 0
        assert cur_target[1] == 0

        dir_to_targ = cur_target - cur_pos

        if self.angular_noise_var > 0:
            angular_noise_rad = np.random.normal(0, self.angular_noise_var)
            while abs(angular_noise_rad) > np.pi:
                angular_noise_rad = np.random.normal(0, self.angular_noise_var)
        else:
            angular_noise_rad = 0
        angle = np.arctan2(dir_to_targ[2], dir_to_targ[0])
        sum_angle = angle + angular_noise_rad
        return self.gain*np.array([np.cos(sum_angle), np.sin(sum_angle)])


class CenterOutCursorGoalJointSpace2D(CenterOutCursorGoal):
    '''2-link arm controller which moves the endpoint toward a target position at a constant speed'''
    def __init__(self, link_lengths, shoulder_anchor, *args, **kwargs):
        '''
        Constructor for CenterOutCursorGoalJointSpace2D

        Parameters
        ----------
        link_lengths: 
        shoulder_anchor: 
        args, kwargs: positional and keyword arguments for parent constructor (CenterOutCursorGoal)


        Returns
        -------
        '''
        self.link_lengths = link_lengths
        self.shoulder_anchor = shoulder_anchor
        super(CenterOutCursorGoalJointSpace2D, self).__init__(*args, **kwargs)


    def get(self, cur_target, cur_pos, keys_pressed=None):
        '''
        cur_target and cur_pos should be specified in workspace coordinates
        '''
        vx, vz = super(CenterOutCursorGoalJointSpace2D, self).get(cur_target, cur_pos, keys_pressed)
        vy = 0

        px, py, pz = cur_pos

        pos = np.array([px, py, pz]) - self.shoulder_anchor
        vel = np.array([vx, vy, vz])

        # Convert to joint velocities
        from riglib.stereo_opengl import ik
        joint_pos, joint_vel = ik.inv_kin_2D(pos, self.link_lengths[0], self.link_lengths[1], vel)
        return joint_vel[0]['sh_vabd'], joint_vel[0]['el_vflex']

class PosFeedbackController(FeedbackController):
    '''
    Dumb controller that just spits back the target
    '''
    def __init__(self, *args, **kwargs):
        pass

    def calc_next_state(self, current_state, target_state, mode=None):
        return target_state

    def __call__(self, current_state, target_state, mode=None):
        return target_state

    def get(self, current_state, target_state, mode=None):
        return target_state