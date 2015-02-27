'''
Various types of "assist", i.e. different methods for shared control
between neural control and machine control. Only applies in cases where
some knowledge of the task goals is available. 
'''

import numpy as np 
from riglib.stereo_opengl import ik
from riglib.bmi import feedback_controllers
import pickle

from state_space_models import StateSpaceArmAssist, StateSpaceReHand, StateSpaceIsMore
from utils.angle_utils import *
from utils.constants import *

class Assister(object):
    '''
    Parent class for various methods of assistive BMI. Children of this class 
    can compute an "optimal" input to the system, which is mixed in with the input
    derived from the subject's neural input. The parent exists primarily for 
    interface standardization and type-checking.
    '''
    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''
        Main assist calculation function

        Parameters
        ----------
        current_state: np.ndarray of shape (n_states, 1)
            Vector representing the current state of the prosthesis 
        target_state: np.ndarray of shape (n_states, 1)
            Vector representing the target state of the prosthesis, i.e. the optimal state for the prosthesis to be in
        assist_level: float
            Number indicating the level of the assist. This can in general have arbitrary units but most assisters
            will have this be a number in the range (0, 1) where 0 is no assist and 1 is full assist
        mode: hashable type, optional, default=None
            Indicator of which mode of the assistive controller to use. When applied, this 'mode' is used as a dictionary key and must be hashable
        kwargs: additional keyword arguments
            These are ignored

        Returns
        -------
        '''
        pass  # implement in subclasses -- should return (Bu, assist_weight)

    def __call__(self, *args, **kwargs):
        '''
        Wrapper for self.calc_assisted_BMI_state
        '''
        return self.calc_assisted_BMI_state(*args, **kwargs)

class LinearFeedbackControllerAssist(Assister):
    '''
    Assister where the machine control is an LQR controller, possibly with different 'modes' depending on the state of the task
    '''
    def __init__(self, A, B, Q, R):
        '''
        Constructor for LinearFeedbackControllerAssist

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
        LinearFeedbackControllerAssist instance
        '''
        self.lqr_controller = feedback_controllers.LQRController(A, B, Q, R)
        # self.A = A
        # self.B = B
        # self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''
        See docs for Assister.calc_assisted_BMI_state
        '''
        Bu = assist_level * self.lqr_controller(current_state, target_state)
        # assist_weight = 0
        # return Bu, assist_weight
        return dict(Bu=Bu, assist_level=0)

class SSMLFCAssister(LinearFeedbackControllerAssist):
    '''
    An LFC assister where the state-space matrices (A, B) are specified from the Decoder's 'ssm' attribute
    '''
    def __init__(self, ssm, Q, R, **kwargs):
        '''
        Constructor for TentacleAssist

        Parameters
        ----------
        ssm: riglib.bmi.state_space_models.StateSpace instance
            The state-space model's A and B matrices represent the system to be controlled
        args: positional arguments
            These are ignored (none are necessary)
        kwargs: keyword arguments
            The constructor must be supplied with the 'kin_chain' kwarg, which must have the attribute 'link_lengths'
            This is specific to 'KinematicChain' plants.

        Returns
        -------
        TentacleAssist instance

        '''        
        if ssm == None:
            raise ValueError("SSMLFCAssister requires a state space model!")

        A, B, W = ssm.get_ssm_matrices()
        super(SSMLFCAssister, self).__init__(A, B, Q, R)
