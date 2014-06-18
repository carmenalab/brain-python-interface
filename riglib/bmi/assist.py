'''
Various types of "assist", i.e. different methods for shared control
between neural control and machine control. Only applies in cases where
some knowledge of the task goals is available. 
'''

import numpy as np 
from riglib.stereo_opengl import ik
from riglib.bmi import feedback_controllers

class Assister(object):
    '''
    Parent class for various methods of assistive BMI. Children of this class 
    can compute an "optimal" input to the system, which is mixed in with the input
    derived from the subject's neural input
    '''
    def __init__(self, *args, **kwargs):
        '''    Docstring    '''
        pass

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''    Docstring    '''
        pass  # implement in subclasses -- should return (Bu, assist_weight)

    def __call__(self, *args, **kwargs):
        '''    Docstring    '''
        return self.calc_assisted_BMI_state(*args, **kwargs)

class LinearFeedbackControllerAssist(Assister):
    '''    Docstring    '''
    def __init__(self, A, B, Q, R):
        '''    Docstring    '''
        self.A = A
        self.B = B
        self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''    Docstring    '''
        B = self.B
        F = self.F
        Bu = assist_level * B*F*(target_state - current_state)
        assist_weight = assist_level
        return Bu, assist_weight

class TentacleAssist(LinearFeedbackControllerAssist):
    '''    Docstring    '''
    def __init__(self, *args, **kwargs):
        '''    Docstring    '''
        kin_chain = kwargs.pop('kin_chain')
        ssm = kwargs.pop('ssm')
        
        A, B, W = ssm.get_ssm_matrices()

        # TODO state dimension is clearly hardcoded below!!!!
        Q = np.mat(np.diag(np.hstack([kin_chain.link_lengths, np.zeros(5)])))
        R = 10000*np.mat(np.eye(B.shape[1]))

        self.A = A
        self.B = B
        self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    def calc_assisted_BMI_state(self, *args, **kwargs):
        '''    Docstring    '''
        Bu, _ = super(TentacleAssist, self).calc_assisted_BMI_state(*args, **kwargs)
        assist_weight = 0
        return Bu, assist_weight


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



## TODO the code below should be a feedback controller equivalent to the "simple" method above
    ## def create_learner(self):
    ##     dt = 0.1
    ##     A = np.mat([[1., 0, 0, dt, 0, 0, 0], 
    ##                 [0., 0, 0, 0,  0, 0, 0],
    ##                 [0., 0, 1, 0, 0, dt, 0],
    ##                 [0., 0, 0, 0, 0,  0, 0],
    ##                 [0., 0, 0, 0, 0,  0, 0],
    ##                 [0., 0, 0, 0, 0,  0, 0],
    ##                 [0., 0, 0, 0, 0,  0, 1]])

    ##     I = np.mat(np.eye(3))
    ##     B = np.vstack([0*I, I, np.zeros([1,3])])
    ##     F_target = np.hstack([I, 0*I, np.zeros([3,1])])
    ##     F_hold = np.hstack([0*I, 0*I, np.zeros([3,1])])
    ##     F_dict = dict(hold=F_hold, target=F_target)
    ##     self.learner = clda.OFCLearner(self.batch_size, A, B, F_dict)
    ##     self.learn_flag = True

