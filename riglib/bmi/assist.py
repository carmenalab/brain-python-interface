'''
Various types of "assist", i.e. different methods for shared control
between neural control and machine control. Only applies in cases where
some knowledge of the task goals is available. 
'''

import numpy as np 
from riglib.stereo_opengl import ik
from riglib.bmi import feedback_controllers

from state_space_models import StateSpaceArmAssist, StateSpaceReHand, StateSpaceIsMore
from utils.angle_utils import *

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
        assist_weight = 0
        # assist_weight = assist_level
        # B = self.B
        # F = self.F
        # Bu = assist_level * B*F*(target_state - current_state)
        # assist_weight = assist_level
        return Bu, assist_weight

class SSMLFCAssister(LinearFeedbackControllerAssist):
    def __init__(ssm, Q, R, **kwargs):
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

        return Bu, assist_weight


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

        return Bu, assist_weight


def endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen=0.1, speed=0.5, target_radius=2., assist_level=0.):
    '''    Docstring    '''
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



####################
## iBMI assisters ##
####################

# simple iBMI assisters

class ArmAssistAssister(Assister):
    '''Simple assister that moves ArmAssist position towards the xy target 
    at a constant speed, and towards the angular target at a constant
    angular speed. When inside the target or close to the angular target,
    these speeds are reduced.'''

    def __init__(self, *args, **kwargs):
        self.decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        self.assist_speed = kwargs.pop('assist_speed', 2.)
        self.target_radius = kwargs.pop('target_radius', 2.)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        if assist_level > 0:
            xy_pos = np.array(current_state[0:2, 0]).ravel()
            target_xy_pos = np.array(target_state[0:2, 0]).ravel()
            assist_xy_pos, assist_xy_vel = self.xy_assist(xy_pos, target_xy_pos)

            psi_pos = np.array(current_state[2, 0]).ravel()
            target_psi_pos = np.array(target_state[2, 0]).ravel()
            assist_psi_pos, assist_psi_vel = self.angle_assist(psi_pos, target_psi_pos)

            # if mode == 'hold':
            #     print 'task state is "hold", setting assist vels to 0'
            #     assist_xy_vel[:] = 0.
            #     assist_psi_vel[:] = 0.

            # print 'assist_xy_vel:', assist_xy_vel

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

        return Bu, assist_weight

    def xy_assist(self, xy_pos, target_xy_pos):
        binlen        = self.decoder_binlen
        speed         = self.assist_speed
        target_radius = self.target_radius 

        diff_vec = target_xy_pos - xy_pos
        dist_to_target = np.linalg.norm(diff_vec)
        dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)

        if dist_to_target > target_radius:
            assist_xy_vel = speed * dir_to_target
        else:
            frac = 0.5 * dist_to_target/target_radius
            assist_xy_vel = frac * speed * dir_to_target

        assist_xy_pos = xy_pos + binlen*assist_xy_vel

        return assist_xy_pos, assist_xy_vel

    def angle_assist(self, ang_pos, target_ang_pos):
        binlen = self.decoder_binlen
        angular_speed = 5*(np.pi/180)  # in rad/s (5 deg/s)
        
        # when angular difference is below cutoff_diff, use smaller angular speeds
        cutoff_diff = 5*(np.pi/180)  # in rad (5 deg)

        ang_diff = angle_subtract(target_ang_pos, ang_pos)
        if abs(ang_diff) > cutoff_diff:
            assist_ang_vel = np.sign(ang_diff) * angular_speed
        else:
            assist_ang_vel = 0.5 * (ang_diff / cutoff_diff) * angular_speed

        assist_ang_pos = ang_pos + assist_ang_vel*binlen

        return assist_ang_pos, assist_ang_vel


class ReHandAssister(Assister):
    '''Simple assister that moves ReHand joint angles towards their angular
    targets at a constant angular speed. When angles are close to the target
    angles, these speeds are reduced.'''

    def __init__(self, *args, **kwargs):
        self.decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        self.assist_speed = kwargs.pop('assist_speed', 5.)
        self.target_radius = kwargs.pop('target_radius', 2.)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        if assist_level > 0:
            assist_rh_pos = np.zeros((0, 1))
            assist_rh_vel = np.zeros((0, 1))

            for i in range(4):
                rh_i_pos = np.array(current_state[i, 0]).ravel()
                target_rh_i_pos = np.array(target_state[i, 0]).ravel()
                assist_rh_i_pos, assist_rh_i_vel = self.angle_assist(rh_i_pos, target_rh_i_pos)
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

        return Bu, assist_weight

    def angle_assist(self, ang_pos, target_ang_pos):
        binlen = self.decoder_binlen
        angular_speed = 5*(np.pi/180)  # in rad/s (5 deg/s)
        
        # when angular difference is below cutoff_diff, use smaller angular speeds
        cutoff_diff = 5*(np.pi/180)  # in rad (5 deg)

        ang_diff = angle_subtract(target_ang_pos, ang_pos)
        if abs(ang_diff) > cutoff_diff:
            assist_ang_vel = np.sign(ang_diff) * angular_speed
        else:
            assist_ang_vel = 0.5 * (ang_diff / cutoff_diff) * angular_speed

        assist_ang_pos = ang_pos + assist_ang_vel*binlen

        return assist_ang_pos, assist_ang_vel


class IsMoreAssister(Assister):
    '''Simple assister that combines an ArmAssistAssister and a 
    ReHandAssister.'''

    def __init__(self, *args, **kwargs):
        self.aa_assister = ArmAssistAssister(*args, **kwargs)
        self.rh_assister = ReHandAssister(*args, **kwargs)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
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

        return Bu, assist_weight


# LFC iBMI assisters

class ArmAssistLFCAssister(LinearFeedbackControllerAssist):
    def __init__(self, *args, **kwargs):
        ssm = StateSpaceArmAssist()
        A, B, _ = ssm.get_ssm_matrices()
        
        # TODO -- velocity cost? not necessary?
        Q = np.mat(np.diag([1., 1., 1., 0, 0, 0, 0]))
        self.Q = Q
        
        R = 1e6 * np.mat(np.diag([1., 1., 1.]))
        self.R = R

        self.A = A
        self.B = B
        self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''Overriding to account for proper subtraction of angles.'''
        B = self.B
        F = self.F

        diff = target_state - current_state
        diff[2] = angle_subtract(target_state[2], current_state[2])

        Bu = assist_level * B*F*(diff)
        # assist_weight = 0
        assist_weight = assist_level

        return Bu, assist_weight


# in progress
# class ArmAssistLFCAssister(SSMLFCAssister):
#     '''Docstring.'''
    
#     def __init__(self, ssm, *args, **kwargs):
#         # '''
#         # Parameters
#         # ----------
#         # ssm: riglib.bmi.state_space_models.StateSpace instance
#         #     The state-space model's A and B matrices represent the system to be controlled
#         # args: positional arguments
#         #     These are ignored (none are necessary)
#         # kwargs: keyword arguments
#         #     The constructor must be supplied with the 'kin_chain' kwarg, which must have the attribute 'link_lengths'
#         #     This is specific to 'KinematicChain' plants.

#         # '''
#         ssm = StateSpaceArmAssist()
#         Q = np.mat(np.diag([1., 1., 1., 0, 0, 0, 0]))        
#         R = 1e6 * np.mat(np.diag([1., 1., 1.]))

#         super(ArmAssistLFCAssister, self).__init__(ssm, Q, R)

#     def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
#         '''Overriding to account for proper subtraction of angles.'''


# class ArmAssistLFCAssister2(LinearFeedbackControllerAssist):
#     def __init__(self, *args, **kwargs):
#         ssm = StateSpaceArmAssist()
#         A, B, _ = ssm.get_ssm_matrices()
        
#         self.A = A

#         B_ = np.vstack([B, np.zeros(3)])
#         self.B_ = B_

#         # TODO -- velocity cost? not necessary?
#         # Q = np.mat(np.diag([1., 1., 1., 0, 0, 0, 0]))
#         Q_ = np.mat(np.diag([1., 1., 1., 0, 0, 0, 0, 0]))
#         self.Q_ = Q_
        
#         R = 1e6 * np.mat(np.diag([1., 1., 1.]))
#         self.R = R

#     def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
#         '''Overriding to account for proper subtraction of angles.'''
#         A = self.A
#         B_ = self.B_
#         Q_ = self.Q_
#         R = self.R

#         A_ = np.vstack([np.hstack([A, A*np.mat(target_state)]), np.array([0, 0, 0, 0, 0, 0, 0, 1])])

#         F_ = feedback_controllers.LQRController.dlqr(A_, B_, Q_, R)

#         diff = target_state - current_state
#         diff[2] = angle_subtract(target_state[2], current_state[2])

#         BF = B_ * F_

#         Bu = assist_level * BF[:-1, :-1] * diff
#         assist_weight = 0

#         return Bu, assist_weight

class ReHandLFCAssister(LinearFeedbackControllerAssist):
    def __init__(self, *args, **kwargs):
        ssm = StateSpaceReHand()
        A, B, _ = ssm.get_ssm_matrices()
        
        # TODO -- velocity cost? not necessary?
        Q = np.mat(np.diag([7., 7., 7., 7., 0, 0, 0, 0, 0]))
        self.Q = Q
        
        R = 1e6 * np.mat(np.diag([1., 1., 1., 1.]))
        self.R = R

        self.A = A
        self.B = B
        self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''Overriding to account for proper subtraction of angles.'''
        B = self.B
        F = self.F

        diff = target_state - current_state
        for i in range(4):
            diff[i] = angle_subtract(target_state[i], current_state[i])

        Bu = assist_level * B*F*(diff)
        assist_weight = 0

        return Bu, assist_weight


class IsMoreLFCAssister(LinearFeedbackControllerAssist):
    def __init__(self, *args, **kwargs):
        ssm = StateSpaceIsMore()
        A, B, _ = ssm.get_ssm_matrices()
        
        # TODO -- velocity cost? not necessary?
        Q = np.mat(np.diag([1., 1., 7., 7., 7., 7., 7., 0, 0, 0, 0, 0, 0, 0, 0]))
        self.Q = Q
        
        R = 1e6 * np.mat(np.diag([1., 1., 1., 1., 1., 1., 1.]))
        self.R = R

        self.A = A
        self.B = B
        self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''Overriding to account for proper subtraction of angles.'''
        B = self.B
        F = self.F

        diff = target_state - current_state
        for i in range(2, 7):
            diff[i] = angle_subtract(target_state[i], current_state[i])

        Bu = assist_level * B*F*(diff)
        assist_weight = 0

        return Bu, assist_weight


class SimpleEndpointAssisterLFC(feedback_controllers.MultiModalLFC):
    def __init__(self, *args, **kwargs):        
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
