'''
Various types of "assist", i.e. different methods for shared control
between neural control and machine control. Only applies in cases where
some knowledge of the task goals is available. 
'''

import numpy as np 
from riglib.stereo_opengl import ik
from riglib.bmi import feedback_controllers

from utils.angle_utils import *

class Assister(object):
    '''
    Parent class for various methods of assistive BMI. Children of this class 
    can compute an "optimal" input to the system, which is mixed in with the input
    derived from the subject's neural input
    '''
    def __init__(self, start_level, end_level, assist_time, assist_speed):
        self.start_level    = start_level
        self.end_level      = end_level
        self.assist_time    = assist_time
        self.assist_speed   = assist_speed

        self.current_level  = start_level

    def update_level(self, task):
        if self.current_level != self.end_level:
            elapsed_time = task.get_time() - task.task_start_time
            temp = self.start_level - elapsed_time*(self.start_level-self.end_level)/self.assist_time
            if temp <= self.end_level:
                self.current_level = self.end_level
                print "Assist reached end level"
            else:
                self.current_level = temp
                if (task.count % 3600 == 0):  # print every minute
                    print "Assist level: ", self.current_level

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        pass  # implement in subclasses -- should return (Bu, assist_weight)

    def __call__(self, *args, **kwargs):
        return self.calc_assisted_BMI_state(*args, **kwargs)

class LinearFeedbackControllerAssist(Assister):
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        B = self.B
        F = self.F
        Bu = assist_level * B*F*(target_state - current_state)
        assist_weight = assist_level
        return Bu, assist_weight

class TentacleAssist(LinearFeedbackControllerAssist):
    def __init__(self, *args, **kwargs):
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
        self.decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        self.assist_speed = kwargs.pop('assist_speed', 5.)
        self.target_radius = kwargs.pop('target_radius', 2.)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
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
    '''Docstring.'''
    def __init__(self, arm, *args, **kwargs):
        self.arm = arm
        super(Joint5DOFEndpointTargetAssister, self).__init__(*args, **kwargs)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
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



class ArmAssistAssister(Assister):
    '''Docstring.'''
    def __init__(self, *args, **kwargs):
        self.decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        self.assist_speed = kwargs.pop('assist_speed', 5.)  # TODO -- 5 cm/s too fast?
        self.target_radius = kwargs.pop('target_radius', 2.)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        if assist_level > 0:
            xy_pos = np.array(current_state[0:2, 0]).ravel()
            target_xy_pos = np.array(target_state[0:2, 0]).ravel()
            assist_xy_pos, assist_xy_vel = self.xy_assist(xy_pos, target_xy_pos)

            psi_pos = np.array(current_state[2, 0]).ravel()
            target_psi_pos = np.array(target_state[2, 0]).ravel()
            assist_psi_pos, assist_psi_vel = self.psi_assist(psi_pos, target_psi_pos)

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
            assist_xy_pos = xy_pos + speed*binlen*dir_to_target
        else:
            frac = dist_to_target/target_radius
            assist_xy_pos = xy_pos + frac*speed*binlen*dir_to_target

        assist_xy_vel = (assist_xy_pos - xy_pos)/binlen

        return assist_xy_pos, assist_xy_vel

    def psi_assist(self, psi_pos, target_psi_pos):
        binlen = self.decoder_binlen

        angular_speed = 15*(np.pi/180)  # in rad/s (15 deg/s)
        cutoff_diff = 5*(np.pi/180)  # in rad (5 deg)
        psi_diff = angle_subtract(target_psi_pos, psi_pos)
        
        if abs(psi_diff) > cutoff_diff:
            assist_psi_vel = np.sign(psi_diff) * angular_speed
        else:
            assist_psi_vel = (psi_diff / cutoff_diff) * angular_speed

        assist_psi_pos = psi_pos + assist_psi_vel*binlen

        return assist_psi_pos, assist_psi_vel


class ReHandAssister(Assister):
    '''Docstring.'''
    def __init__(self, *args, **kwargs):
        self.decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        self.assist_speed = kwargs.pop('assist_speed', 5.)
        self.target_radius = kwargs.pop('target_radius', 2.)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        if assist_level > 0:
            # TODO -- code below is *unnecessarily* long/inefficient, find way to shorten
            rh1_pos = np.array(current_state[0, 0]).ravel()
            target_rh1_pos = np.array(target_state[0, 0]).ravel()
            assist_rh1_pos, assist_rh1_vel = self.angle_assist(rh1_pos, target_rh1_pos)

            rh2_pos = np.array(current_state[1, 0]).ravel()
            target_rh2_pos = np.array(target_state[1, 0]).ravel()
            assist_rh2_pos, assist_rh2_vel = self.angle_assist(rh2_pos, target_rh2_pos)

            rh3_pos = np.array(current_state[2, 0]).ravel()
            target_rh3_pos = np.array(target_state[2, 0]).ravel()
            assist_rh3_pos, assist_rh3_vel = self.angle_assist(rh3_pos, target_rh3_pos)

            rh4_pos = np.array(current_state[3, 0]).ravel()
            target_rh4_pos = np.array(target_state[3, 0]).ravel()
            assist_rh4_pos, assist_rh4_vel = self.angle_assist(rh4_pos, target_rh4_pos)

            Bu = assist_level * np.hstack([assist_rh1_pos,
                                           assist_rh2_pos,
                                           assist_rh3_pos,
                                           assist_rh4_pos,
                                           assist_rh1_vel,
                                           assist_rh2_vel,
                                           assist_rh3_vel,
                                           assist_rh4_vel,
                                           1])

            Bu = np.mat(Bu.reshape(-1, 1))

            assist_weight = assist_level
        else:
            Bu = None
            assist_weight = 0.

        return Bu, assist_weight

    def angle_assist(self, ang_pos, target_ang_pos):
        binlen = self.decoder_binlen

        angular_speed = 15*(np.pi/180)  # in rad/s (15 deg/s)
        cutoff_diff = 5*(np.pi/180)  # in rad (5 deg)
        ang_diff = angle_subtract(target_ang_pos, ang_pos)
        
        if abs(ang_diff) > cutoff_diff:
            assist_ang_vel = np.sign(ang_diff) * angular_speed
        else:
            assist_ang_vel = (ang_diff / cutoff_diff) * angular_speed

        assist_ang_pos = ang_pos + assist_ang_vel*binlen

        return assist_ang_pos, assist_ang_vel


class IsMoreAssister(Assister):
    '''Docstring.'''
    def __init__(self, *args, **kwargs):
        self.decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        self.assist_speed = kwargs.pop('assist_speed', 5.)
        self.target_radius = kwargs.pop('target_radius', 2.)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        if assist_level > 0:
            # TODO -- code below is *unnecessarily* long/inefficient, find way to shorten

            # ArmAssist
            xy_pos = np.array(current_state[0:2, 0]).ravel()
            target_xy_pos = np.array(target_state[0:2, 0]).ravel()
            assist_xy_pos, assist_xy_vel = self.xy_assist(xy_pos, target_xy_pos)

            psi_pos = np.array(current_state[2, 0]).ravel()
            target_psi_pos = np.array(target_state[2, 0]).ravel()
            assist_psi_pos, assist_psi_vel = self.angle_assist(psi_pos, target_psi_pos)

            # ReHand
            rh1_pos = np.array(current_state[3, 0]).ravel()
            target_rh1_pos = np.array(target_state[3, 0]).ravel()
            assist_rh1_pos, assist_rh1_vel = self.angle_assist(rh1_pos, target_rh1_pos)

            rh2_pos = np.array(current_state[4, 0]).ravel()
            target_rh2_pos = np.array(target_state[4, 0]).ravel()
            assist_rh2_pos, assist_rh2_vel = self.angle_assist(rh2_pos, target_rh2_pos)

            rh3_pos = np.array(current_state[5, 0]).ravel()
            target_rh3_pos = np.array(target_state[5, 0]).ravel()
            assist_rh3_pos, assist_rh3_vel = self.angle_assist(rh3_pos, target_rh3_pos)

            rh4_pos = np.array(current_state[6, 0]).ravel()
            target_rh4_pos = np.array(target_state[6, 0]).ravel()
            assist_rh4_pos, assist_rh4_vel = self.angle_assist(rh4_pos, target_rh4_pos)

            Bu = assist_level * np.hstack([assist_xy_pos, 
                                           assist_psi_pos,
                                           assist_rh1_pos,
                                           assist_rh2_pos,
                                           assist_rh3_pos,
                                           assist_rh4_pos,
                                           assist_xy_vel,
                                           assist_psi_vel,
                                           assist_rh1_vel,
                                           assist_rh2_vel,
                                           assist_rh3_vel,
                                           assist_rh4_vel,
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
            assist_xy_pos = xy_pos + speed*binlen*dir_to_target
        else:
            frac = dist_to_target/target_radius
            assist_xy_pos = xy_pos + frac*speed*binlen*dir_to_target

        assist_xy_vel = (assist_xy_pos - xy_pos)/binlen

        return assist_xy_pos, assist_xy_vel

    def angle_assist(self, ang_pos, target_ang_pos):
        binlen = self.decoder_binlen

        angular_speed = 15*(np.pi/180)  # in rad/s (15 deg/s)
        cutoff_diff = 5*(np.pi/180)  # in rad (5 deg)
        ang_diff = angle_subtract(target_ang_pos, ang_pos)
        
        if abs(ang_diff) > cutoff_diff:
            assist_ang_vel = np.sign(ang_diff) * angular_speed
        else:
            assist_ang_vel = (ang_diff / cutoff_diff) * angular_speed

        assist_ang_pos = ang_pos + assist_ang_vel*binlen

        return assist_ang_pos, assist_ang_vel



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

