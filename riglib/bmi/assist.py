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


class OFCEndpointAssister(Assister):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, decoding_rate=180):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.F_assist = pickle.load(open('/storage/assist_params/assist_20levels_ppf.pkl'))
        self.n_assist_levels = len(self.F_assist)                              
        self.prev_assist_level = self.n_assist_levels          
        self.B = np.mat(np.vstack([np.zeros([3,3]), np.eye(3)*1000*1./decoding_rate, np.zeros(3)]))

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        ##assist_level_idx = min(int(assist_level * self.n_assist_levels), self.n_assist_levels-1)
        ##if assist_level_idx < self.prev_assist_level:                        
        ##    print "assist_level_idx decreasing to", assist_level_idx         
        ##    self.prev_assist_level = assist_level_idx                        
        ##F = np.mat(self.F_assist[assist_level_idx])    
        F = self.get_F(assist_level)
        Bu = self.B*F*(target_state - current_state)
        return dict(F=F, x_target=target_state)
        # print Bu
        # return Bu, 0

    def get_F(self, assist_level):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        assist_level_idx = min(int(assist_level * self.n_assist_levels), self.n_assist_levels-1)
        # if assist_level_idx < self.prev_assist_level:                        
        #     print "assist_level_idx decreasing to", assist_level_idx         
        #     self.prev_assist_level = assist_level_idx                        
        F = np.mat(self.F_assist[assist_level_idx])    
        return F

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
    An LFC assister where the state-space matrices (A, B) are specified based on 
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

class TentacleAssist(SSMLFCAssister):
    '''
    Assister which can be used for a kinematic chain of any length. The cost function is calibrated for the experiments with the 4-link arm
    '''
    def __init__(self, ssm, *args, **kwargs):
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
        try:
            kin_chain = kwargs.pop('kin_chain')
        except KeyError:
            raise ValueError("kin_chain must be supplied for TentacleAssist")
        
        update_rate = kwargs.pop('update_rate', 0.1)
        A, B, W = ssm.get_ssm_matrices(update_rate=update_rate)
        Q = np.mat(np.diag(np.hstack([kin_chain.link_lengths, np.zeros_like(kin_chain.link_lengths), 0])))
        R = 10000*np.mat(np.eye(B.shape[1]))

        super(TentacleAssist, self).__init__(ssm, Q, R)

    # def calc_assisted_BMI_state(self, *args, **kwargs):
    #     '''
    #     see Assister.calc_assisted_BMI_state. This method always returns an 'assist_weight' of 0, 
    #     which is required for the feedback controller style of assist to cooperate with the rest of the 
    #     Decoder
    #     '''
    #     Bu, _ = super(TentacleAssist, self).calc_assisted_BMI_state(*args, **kwargs)
    #     assist_weight = 0
    #     return Bu, assist_weight


class SimpleEndpointAssister(Assister):
    '''
    Constant velocity toward the target if the cursor is outside the target. If the
    cursor is inside the target, the speed becomes the distance to the center of the
    target divided by 2.
    '''
    def __init__(self, *args, **kwargs):
        '''    Docstring    '''
        self.decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        self.assist_speed = kwargs.pop('assist_speed', 5.)
        self.target_radius = kwargs.pop('target_radius', 2.)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''    Docstring    '''
        Bu = None
        assist_weight = 0.

        if assist_level > 0:
            cursor_pos = np.array(current_state[0:3,0]).ravel()
            target_pos = np.array(target_state[0:3,0]).ravel()
            decoder_binlen = self.decoder_binlen
            speed = self.assist_speed * decoder_binlen
            target_radius = self.target_radius
            Bu = endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen, speed, target_radius, assist_level)
            assist_weight = assist_level 

        # return Bu, assist_weight
        return dict(Bu=Bu, assist_level=assist_weight)


class Joint5DOFEndpointTargetAssister(SimpleEndpointAssister):
    '''
    Assister for 5DOF 3-D arm (e.g., a kinematic model of the exoskeleton), restricted to movements in a 2D plane
    '''
    def __init__(self, arm, *args, **kwargs):
        '''    Docstring    '''
        self.arm = arm
        super(Joint5DOFEndpointTargetAssister, self).__init__(*args, **kwargs)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''    Docstring    '''
        Bu = None # By default, no assist
        assist_weight = 0.

        if assist_level> 0:
            cursor_joint_pos = np.asarray(current_state)[[1,3],0]
            cursor_pos       = self.arm.perform_fk(cursor_joint_pos)
            target_joint_pos = np.asarray(target_state)[[1,3],0]
            target_pos       = self.arm.perform_fk(target_joint_pos)

            arm              = self.arm
            decoder_binlen   = self.decoder_binlen
            speed            = self.assist_speed * decoder_binlen
            target_radius    = self.target_radius

            # Get the endpoint control under full assist
            # Note: the keyword argument "assist_level" is intended to be set to 1. (and not self.current_level) 
            Bu_endpoint = endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen, speed, target_radius, assist_level=1.)

            # Convert the endpoint assist to joint space using IK/Jacobian
            Bu_endpoint = np.array(Bu_endpoint).ravel()
            endpt_pos = Bu_endpoint[0:3]
            endpt_vel = Bu_endpoint[3:6]

            l_upperarm, l_forearm = arm.link_lengths
            shoulder_center = np.array([0., 0., 0.])#arm.xfm.move
            joint_pos, joint_vel = ik.inv_kin_2D(endpt_pos - shoulder_center, l_upperarm, l_forearm, vel=endpt_vel)

            Bu_joint = np.hstack([joint_pos[0].view((np.float64, 5)), joint_vel[0].view((np.float64, 5)), 1]).reshape(-1, 1)

            # Downweight the joint assist
            Bu = assist_level * np.mat(Bu_joint).reshape(-1,1)
            assist_weight = assist_level

        return dict(Bu=Bu, assist_level=assist_weight)
        # return Bu, assist_weight


def endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen=0.1, speed=0.5, target_radius=2., assist_level=0.):
    '''
    Parameters
    ----------
    cursor_pos: np.ndarray of shape (3,)
        Current position of the cursor
    target_pos: np.ndarray of shape (3,)
        Specified target position
    decoder_binlen: float
        Time between iterations of the decoder
    speed: float
        Speed of the machine-assisted cursor
    target_radius: float
        Radius of the target. When the cursor is inside the target, the machine assisted cursor speed decreases.
    assist_level: float
        Scalar between (0, 1) where 1 indicates full machine control and 0 indicates full neural control.

    Returns
    -------
    Bu: np.ndarray of shape (7, 1)
        Control vector to add onto the state vector to assist control.
    '''
    diff_vec = target_pos - cursor_pos 
    dist_to_target = np.linalg.norm(diff_vec)
    dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)
    
    if dist_to_target > target_radius:
        assist_cursor_pos = cursor_pos + speed*dir_to_target
    else:
        assist_cursor_pos = cursor_pos + speed*diff_vec/2

    assist_cursor_vel = (assist_cursor_pos-cursor_pos)/decoder_binlen
    Bu = assist_level * np.hstack([assist_cursor_pos, assist_cursor_vel, 1])
    Bu = np.mat(Bu.reshape(-1,1))
    return Bu


class SimpleEndpointAssisterLFC(feedback_controllers.MultiModalLFC):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, *args, **kwargs):        
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        dt = 0.1
        A = np.mat([[1., 0, 0, dt, 0, 0, 0], 
                    [0., 1, 0, 0,  dt, 0, 0],
                    [0., 0, 1, 0, 0, dt, 0],
                    [0., 0, 0, 0, 0,  0, 0],
                    [0., 0, 0, 0, 0,  0, 0],
                    [0., 0, 0, 0, 0,  0, 0],
                    [0., 0, 0, 0, 0,  0, 1]])

        I = np.mat(np.eye(3))
        B = np.vstack([0*I, I, np.zeros([1,3])])
        F_target = np.hstack([I, 0*I, np.zeros([3,1])])
        F_hold = np.hstack([0*I, 0*I, np.zeros([3,1])])
        F_dict = dict(hold=F_hold, target=F_target)
        super(SimpleEndpointAssisterLFC, self).__init__(B=B, F=F_dict)



####################
## iBMI assisters ##
####################

# simple iBMI assisters

class ArmAssistAssister(Assister):
    '''Simple assister that moves ArmAssist position towards the xy target 
    at a constant speed, and towards the psi target at a constant angular 
    speed. When within a certain xy distance or angular distance of the 
    target, these speeds are reduced.'''

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.call_rate  = kwargs.pop('call_rate',  10)             # secs
        self.xy_speed   = kwargs.pop('xy_speed',   2.)             # cm/s
        self.xy_cutoff  = kwargs.pop('xy_cutoff',  2.)             # cm
        self.psi_speed  = kwargs.pop('psi_speed',  5.*deg_to_rad)  # rad/s
        self.psi_cutoff = kwargs.pop('psi_cutoff', 5.*deg_to_rad)  # rad

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if assist_level > 0:
            xy_pos  = np.array(current_state[0:2, 0]).ravel()
            psi_pos = np.array(current_state[  2, 0]).ravel()
            target_xy_pos  = np.array(target_state[0:2, 0]).ravel()
            target_psi_pos = np.array(target_state[  2, 0]).ravel()
            assist_xy_pos, assist_xy_vel = self._xy_assist(xy_pos, target_xy_pos)
            assist_psi_pos, assist_psi_vel = self._psi_assist(psi_pos, target_psi_pos)

            # if mode == 'hold':
            #     print 'task state is "hold", setting assist vels to 0'
            #     assist_xy_vel[:] = 0.
            #     assist_psi_vel[:] = 0.

            Bu = assist_level * np.hstack([assist_xy_pos, 
                                           assist_psi_pos,
                                           assist_xy_vel,
                                           assist_psi_vel,
                                           1])
            Bu = np.mat(Bu.reshape(-1, 1))

            assist_weight = assist_level
        else:
            Bu = None
            assist_weight = 0.
        return dict(Bu=Bu, assist_level=assist_weight)
        # return Bu, assist_weight

    def _xy_assist(self, xy_pos, target_xy_pos):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        diff_vec = target_xy_pos - xy_pos
        dist_to_target = np.linalg.norm(diff_vec)
        dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)

        # if xy distance is below xy_cutoff (e.g., target radius), use smaller speed
        if dist_to_target < self.xy_cutoff:
            frac = 0.5 * dist_to_target / self.xy_cutoff
            assist_xy_vel = frac * self.xy_speed * dir_to_target
        else:
            assist_xy_vel = self.xy_speed * dir_to_target

        assist_xy_pos = xy_pos + assist_xy_vel/self.call_rate

        return assist_xy_pos, assist_xy_vel

    def _psi_assist(self, psi_pos, target_psi_pos):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        psi_diff = angle_subtract(target_psi_pos, psi_pos)

        # if angular distance is below psi_cutoff, use smaller speed
        if abs(psi_diff) < self.psi_cutoff:
            assist_psi_vel = 0.5 * (psi_diff / self.psi_cutoff) * self.psi_speed
        else:
            assist_psi_vel = np.sign(psi_diff) * self.psi_speed

        assist_psi_pos = psi_pos + assist_psi_vel/self.call_rate

        return assist_psi_pos, assist_psi_vel


class ReHandAssister(Assister):
    '''Simple assister that moves ReHand joint angles towards their angular
    targets at a constant angular speed. When angles are close to the target
    angles, these speeds are reduced.'''

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.call_rate  = kwargs.pop('call_rate' , 10)             # secs
        self.ang_speed  = kwargs.pop('ang_speed',  5.*deg_to_rad)  # rad/s
        self.ang_cutoff = kwargs.pop('ang_cutoff', 5.*deg_to_rad)  # rad

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if assist_level > 0:
            assist_rh_pos = np.zeros((0, 1))
            assist_rh_vel = np.zeros((0, 1))

            for i in range(4):
                rh_i_pos = np.array(current_state[i, 0]).ravel()
                target_rh_i_pos = np.array(target_state[i, 0]).ravel()
                assist_rh_i_pos, assist_rh_i_vel = self._angle_assist(rh_i_pos, target_rh_i_pos)
                assist_rh_pos = np.vstack([assist_rh_pos, assist_rh_i_pos])
                assist_rh_vel = np.vstack([assist_rh_vel, assist_rh_i_vel])

            # if mode == 'hold':
            #     print 'task state is "hold", setting assist vels to 0'
            #     assist_rh_vel[:] = 0.

            Bu = assist_level * np.vstack([assist_rh_pos,
                                           assist_rh_vel,
                                           1])
            Bu = np.mat(Bu.reshape(-1, 1))

            assist_weight = assist_level
        else:
            Bu = None
            assist_weight = 0.
        return dict(Bu=Bu, assist_level=assist_weight)
        # return Bu, assist_weight

    def _angle_assist(self, ang_pos, target_ang_pos):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        ang_diff = angle_subtract(target_ang_pos, ang_pos)
        if abs(ang_diff) > self.ang_cutoff:
            assist_ang_vel = np.sign(ang_diff) * self.ang_speed
        else:
            assist_ang_vel = 0.5 * (ang_diff / self.ang_cutoff) * self.ang_speed

        assist_ang_pos = ang_pos + assist_ang_vel/self.call_rate

        return assist_ang_pos, assist_ang_vel


class IsMoreAssister(Assister):
    '''Combines an ArmAssistAssister and a ReHandAssister.'''

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.aa_assister = ArmAssistAssister(*args, **kwargs)
        self.rh_assister = ReHandAssister(*args, **kwargs)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if assist_level > 0:
            aa_current_state = np.vstack([current_state[0:3], current_state[7:10], 1])
            aa_target_state  = np.vstack([target_state[0:3], target_state[7:10], 1])
            aa_Bu = self.aa_assister.calc_assisted_BMI_state(aa_current_state,
                                                             aa_target_state,
                                                             assist_level,
                                                             mode=mode,
                                                             **kwargs)[0]

            rh_current_state = np.vstack([current_state[3:7], current_state[10:14], 1])
            rh_target_state  = np.vstack([target_state[3:7], target_state[10:14], 1])
            rh_Bu = self.rh_assister.calc_assisted_BMI_state(rh_current_state,
                                                             rh_target_state,
                                                             assist_level,
                                                             mode=mode,
                                                             **kwargs)[0]

            Bu = np.vstack([aa_Bu[0:3],
                            rh_Bu[0:4],
                            aa_Bu[3:6],
                            rh_Bu[4:8],
                            assist_level * 1])
            Bu = np.mat(Bu.reshape(-1, 1))

            assist_weight = assist_level
        else:
            Bu = None
            assist_weight = 0.
        return dict(Bu=Bu, assist_level=assist_weight)
        # return Bu, assist_weight


# LFC iBMI assisters
# not inheriting from LinearFeedbackControllerAssist/SSMLFCAssister because:
# - use of "special angle subtraction" when doing target_state - current_state
# - meant to be used with 'weighted_avg_lfc'=True decoder kwarg, and thus 
#   assist_weight is set to assist_level, not to 0

class ArmAssistLFCAssister(Assister):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        ssm = StateSpaceArmAssist()
        A, B, _ = ssm.get_ssm_matrices()
        Q = np.mat(np.diag([1., 1., 1., 0, 0, 0, 0]))
        R = 1e6 * np.mat(np.diag([1., 1., 1.]))

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''TODO.'''

        diff = target_state - current_state
        diff[2] = angle_subtract(target_state[2], current_state[2])

        Bu = assist_level * self.B*self.F*diff
        assist_weight = assist_level
        return dict(Bu=Bu, assist_level=assist_weight)
        # return Bu, assist_weight


class ReHandLFCAssister(Assister):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        ssm = StateSpaceReHand()
        A, B, _ = ssm.get_ssm_matrices()
        Q = np.mat(np.diag([1., 1., 1., 1., 0, 0, 0, 0, 0]))
        R = 1e6 * np.mat(np.diag([1., 1., 1., 1.]))
        
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''TODO.'''

        diff = target_state - current_state
        for i in range(4):
            diff[i] = angle_subtract(target_state[i], current_state[i])

        Bu = assist_level * self.B*self.F*diff
        assist_weight = assist_level
        return dict(Bu=Bu, assist_level=assist_weight)
        # return Bu, assist_weight


class IsMoreLFCAssister(Assister):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        ssm = StateSpaceIsMore()
        A, B, _ = ssm.get_ssm_matrices()        
        Q = np.mat(np.diag([1., 1., 7., 7., 7., 7., 7., 0, 0, 0, 0, 0, 0, 0, 0]))
        R = 1e6 * np.mat(np.diag([1., 1., 1., 1., 1., 1., 1.]))

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''TODO.'''
        diff = target_state - current_state
        for i in range(2, 7):
            diff[i] = angle_subtract(target_state[i], current_state[i])

        Bu = assist_level * self.B*self.F*diff
        assist_weight = assist_level
        return dict(Bu=Bu, assist_level=assist_weight)
        # return Bu, assist_weight
