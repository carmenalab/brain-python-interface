import numpy as np 
from riglib.stereo_opengl import ik


class Assister(object): # should be an abstract class
    '''Docstring.'''
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

    def calc_assisted_BMI_state(self, task):
        pass  # implement in subclasses -- should return (Bu, assist_weight)


class SimpleEndpointAssister(Assister):
    '''Docstring.'''
    def __init__(self, *args, **kwargs):
        super(SimpleEndpointAssister, self).__init__(*args, **kwargs)
        print 'simple endpoint assister created'

    def calc_assisted_BMI_state(self, task):
        Bu = None # By default, no assist
        assist_weight = 0.

        if self.current_level > 0:
            cursor_pos      = task.decoder['hand_px', 'hand_py', 'hand_pz']
            target_pos      = task.target_location
            decoder_binlen  = task.decoder.binlen
            speed           = self.assist_speed * decoder_binlen
            target_radius   = task.target_radius

            Bu = endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen, speed, target_radius, self.current_level)
            assist_weight = self.current_level

        return Bu, assist_weight


class Joint5DOFEndpointTargetAssister(Assister):
    '''Docstring.'''
    def __init__(self, *args, **kwargs):
        super(Joint5DOFEndpointTargetAssister, self).__init__(*args, **kwargs)

    def calc_assisted_BMI_state(self, task):
        Bu = None # By default, no assist
        assist_weight = 0.

        if self.current_level > 0:
            cursor_pos      = task.get_arm_endpoint()
            target_pos      = task.target_location
            arm             = task.arm
            decoder_binlen  = task.decoder.binlen
            speed           = self.assist_speed * decoder_binlen
            target_radius   = task.target_radius

            # Get the endpoint control under full assist
            # Note: the keyword argument "assist_level" is intended to be set to 1. (and not self.current_level) 
            Bu_endpoint = endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen, speed, target_radius, assist_level=1.)

            # Convert the endpoint assist to joint space using IK/Jacobian
            Bu_endpoint = np.array(Bu_endpoint).ravel()
            endpt_pos = Bu_endpoint[0:3]
            endpt_vel = Bu_endpoint[3:6]

            # TODO when the arm configuration changes, these need to be switched!
            l_forearm, l_upperarm = arm.link_lengths
            shoulder_center = arm.xfm.move
            joint_pos, joint_vel = ik.inv_kin_2D(endpt_pos - shoulder_center, l_upperarm, l_forearm, vel=endpt_vel)

            Bu_joint = np.hstack([joint_pos[0].view((np.float64, 5)), joint_vel[0].view((np.float64, 5)), 1]).reshape(-1, 1)

            # Downweight the joint assist
            Bu = self.current_level * np.mat(Bu_joint).reshape(-1,1)
            assist_weight = self.current_level

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

