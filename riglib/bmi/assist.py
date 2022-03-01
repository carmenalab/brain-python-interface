'''
Various types of "assist", i.e. different methods for shared control
between neural control and machine control. Only applies in cases where
some knowledge of the task goals is available. 
'''

import numpy as np 
from ..bmi import feedback_controllers
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
        pass

    def __call__(self, *args, **kwargs):
        '''
        Wrapper for self.calc_assisted_BMI_state
        '''
        return self.calc_assisted_BMI_state(*args, **kwargs)

class FeedbackControllerAssist(Assister):
    '''
    Assister where the machine control is an LQR controller, possibly with different 'modes' depending on the state of the task
    '''
    def __init__(self, fb_ctrl, style='additive'):
        '''
        Parameters
        ----------
        fb_ctrl : feedback_controllers.FeedbackController instance
            TODO

        Returns
        -------
        FeedbackControllerAssist instance
        '''
        self.fb_ctrl = fb_ctrl
        self.style = style
        assert self.style in ['additive', 'mixing', 'additive_cov']

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''
        See docs for Assister.calc_assisted_BMI_state
        '''
        if self.style == 'additive':
            Bu = assist_level * self.fb_ctrl(current_state, target_state, mode=mode)
            return dict(Bu=Bu, assist_level=0)
        elif self.style == 'mixing':
            x_assist = self.fb_ctrl.calc_next_state(current_state, target_state, mode=mode)
            return dict(x_assist=x_assist, assist_level=assist_level)
        elif self.style == 'additive_cov':
            F = self.get_F(assist_level)
            return dict(F=F, x_target=target_state)            

class FeedbackControllerAssist_StateSpecAssistLevels(FeedbackControllerAssist):
    '''
    Assister where machine controller is LQR controller, but different assist_levels for 
    different control variables (e.g. X,Y,PSI in ArmAssist vs. Rehand)
    '''
    def __init__(self, fb_ctrl, style='additive', **kwargs):
        super(FeedbackControllerAssist_StateSpecAssistLevels, self).__init__(fb_ctrl, style)
        
        # Currently this assister assumes that plant is IsMore Plant: 
        self.assist_level_state_ix = dict()
        self.assist_level_state_ix[0] = np.array([0, 1, 2, 7, 8, 9]) # ARM ASSIST
        self.assist_level_state_ix[1] = np.array([3, 4, 5, 6, 10, 11, 12, 13]) # REHAND
        

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        if self.style == 'additive':
            Bu = self.fb_ctrl(current_state, target_state, mode=mode)
            for ia, al in enumerate(assist_level):
                Bu[self.assist_level_state_ix[ia]] = al*Bu[self.assist_level_state_ix[ia]]
            return dict(Bu=Bu, assist_level=0)
        
        elif self.style == 'mixing':
            x_assist = self.fb_ctrl.calc_next_state(current_state, target_state, mode=mode)
            return dict(x_assist=x_assist, assist_level=assist_level, assist_level_ix=self.assist_level_state_ix)
        

class SSMLFCAssister(FeedbackControllerAssist):
    '''
    An LFC assister where the state-space matrices (A, B) are specified from the Decoder's 'ssm' attribute
    '''
    def __init__(self, ssm, Q, R, **kwargs):
        '''
        Constructor for SSMLFCAssister

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
        SSMLFCAssister instance

        '''        
        if ssm is None:
            raise ValueError("SSMLFCAssister requires a state space model!")

        A, B, W = ssm.get_ssm_matrices()
        self.lqr_controller = feedback_controllers.LQRController(A, B, Q, R)
