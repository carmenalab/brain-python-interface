'''
BMI tasks in the new structure, i.e. inheriting from manualcontrolmultitasks
'''
import numpy as np
import pickle

from riglib.experiment import traits

from riglib.bmi import goal_calculators, ppfdecoder, feedback_controllers
from riglib.bmi.bmi import BMILoop
from riglib.bmi.assist import Assister, FeedbackControllerAssist
from riglib.bmi.state_space_models import StateSpaceEndptVel2D, StateSpaceNLinkPlanarChain

from riglib.stereo_opengl.window import WindowDispl2D
from .target_capture_task import ScreenReachAngle, ScreenTargetCapture
from features.bmi_task_features import LinearlyDecreasingAssist
from .target_graphics import target_colors

np.set_printoptions(suppress=False)

###################
####### Assisters
##################
class OFCEndpointAssister(FeedbackControllerAssist):
    '''
    Assister for cursor PPF control which uses linear feedback (infinite horizon LQR) to drive the cursor toward the target state
    '''
    def __init__(self, decoding_rate=180):
        '''
        Constructor for OFCEndpointAssister

        Parameters
        ----------
        decoding_rate : int
            Rate that the decoder should operate, in Hz. Should be a multiple or divisor of 60 Hz

        Returns
        -------
        OFCEndpointAssister instance
        '''
        F_dict = pickle.load(open('/storage/assist_params/assist_20levels_ppf.pkl'))
        B = np.mat(np.vstack([np.zeros([3,3]), np.eye(3)*1000*1./decoding_rate, np.zeros(3)]))
        fb_ctrl = feedback_controllers.MultiModalLFC(A=B, B=B, F_dict=F_dict)
        super(OFCEndpointAssister, self).__init__(fb_ctrl, style='additive_cov')
        self.n_assist_levels = len(F_dict)

    def get_F(self, assist_level):
        '''
        Look up the feedback gain matrix based on the assist_level

        Parameters
        ----------
        assist_level : float
            Float between 0 and 1 to indicate the level of the assist (1 being the highest)

        Returns
        -------
        np.mat
        '''
        assist_level_idx = min(int(assist_level * self.n_assist_levels), self.n_assist_levels-1)
        F = np.mat(self.fb_ctrl.F_dict[assist_level_idx])
        return F

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
            Bu = self.endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen, speed, target_radius, assist_level)
            assist_weight = assist_level

        # return Bu, assist_weight
        return dict(x_assist=Bu, assist_level=assist_weight)

    @staticmethod
    def endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen=0.1, speed=0.5, target_radius=2., assist_level=0.):
        '''
        Estimate the next state using a constant velocity estimate moving toward the specified target

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
        x_assist : np.ndarray of shape (7, 1)
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
        x_assist = np.hstack([assist_cursor_pos, assist_cursor_vel, 1])
        x_assist = np.mat(x_assist.reshape(-1,1))
        return x_assist

class SimplePosAssister(SimpleEndpointAssister):
    
    @staticmethod 
    def endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen=0.1, speed=0.5, target_radius=2., assist_level=0.):
        '''
        Estimate the next state using a constant velocity estimate moving toward the specified target

        Parameters
        ----------
        see SimpleEndtpointAssister for docs

        Returns
        -------
        x_assist : np.ndarray of shape (7, 1)
            Control vector to add onto the state vector to assist control.
        '''
        diff_vec = target_pos - cursor_pos 
        dist_to_target = np.linalg.norm(diff_vec)
        dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)
        
        if dist_to_target > target_radius:
            assist_cursor_pos = cursor_pos + speed*dir_to_target
        else:
            assist_cursor_pos = cursor_pos + speed*diff_vec/2

        return assist_cursor_pos.ravel()

class SimpleEndpointAssisterLFC(feedback_controllers.MultiModalLFC):
    '''
    Docstring
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
        super(SimpleEndpointAssisterLFC, self).__init__(B=B, F_dict=F_dict)


#################
##### Tasks #####
#################
class BMIControlMultiMixin(BMILoop, LinearlyDecreasingAssist):
    '''
    Target capture task with cursor position controlled by BMI output.
    Cursor movement can be assisted toward target by setting assist_level > 0.
    '''

    background = (.5,.5,.5,1) # Set the screen background color to grey
    reset = traits.Int(0, desc='reset the decoder state to the starting configuration')
    cursor_color = traits.OptionsList("orange", *target_colors, desc='Color of cursor endpoint', bmi3d_input_options=list(target_colors.keys()))

    static_states = ['reward']

    ordered_traits = ['session_length', 'assist_level', 'assist_level_time', 'reward_time','timeout_time','timeout_penalty_time']
    exclude_parent_traits = ['marker_count', 'marker_num', 'goal_cache_block']

    static_states = [] # states in which the decoder is not run
    hidden_traits = ['arm_hide_rate', 'arm_visible', 'hold_penalty_time', 'rand_start', 'reset', 'target_radius', 'window_size']

    is_bmi_seed = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def create_assister(self):
        # Create the appropriate type of assister object
        start_level, end_level = self.assist_level
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        if hasattr(self, 'assist_speed'):
            kwargs['assist_speed'] = self.assist_speed

        if isinstance(self.decoder.ssm, StateSpaceEndptVel2D) and isinstance(self.decoder, ppfdecoder.PPFDecoder):
            self.assister = OFCEndpointAssister()
        elif isinstance(self.decoder.ssm, StateSpaceEndptVel2D):
            self.assister = SimpleEndpointAssister(**kwargs)
        ## elif (self.decoder.ssm == namelist.tentacle_2D_state_space) or (self.decoder.ssm == namelist.joint_2D_state_space):
        ##     # kin_chain = self.plant.kin_chain
        ##     # A, B, W = self.decoder.ssm.get_ssm_matrices(update_rate=self.decoder.binlen)
        ##     # Q = np.mat(np.diag(np.hstack([kin_chain.link_lengths, np.zeros_like(kin_chain.link_lengths), 0])))
        ##     # R = 10000*np.mat(np.eye(B.shape[1]))

        ##     # fb_ctrl = LQRController(A, B, Q, R)
        ##     # self.assister = FeedbackControllerAssist(fb_ctrl, style='additive')
        ##     self.assister = TentacleAssist(ssm=self.decoder.ssm, kin_chain=self.plant.kin_chain, update_rate=self.decoder.binlen)
        else:
            raise NotImplementedError("Cannot assist for this type of statespace: %r" % self.decoder.ssm)

        print(self.assister)

    def create_goal_calculator(self):
        if isinstance(self.decoder.ssm, StateSpaceEndptVel2D):
            self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)
        elif isinstance(self.decoder.ssm, StateSpaceNLinkPlanarChain) and self.decoder.ssm.n_links == 2:
            self.goal_calculator = goal_calculators.PlanarMultiLinkJointGoal(self.decoder.ssm, self.plant.base_loc, self.plant.kin_chain, multiproc=False, init_resp=None)
        elif isinstance(self.decoder.ssm, StateSpaceNLinkPlanarChain) and self.decoder.ssm.n_links == 4:
            shoulder_anchor = self.plant.base_loc
            chain = self.plant.kin_chain
            q_start = self.plant.get_intrinsic_coordinates()
            x_init = np.hstack([q_start, np.zeros_like(q_start), 1])
            x_init = np.mat(x_init).reshape(-1, 1)

            cached = True

            if cached:
                goal_calc_class = goal_calculators.PlanarMultiLinkJointGoalCached
                multiproc = False
            else:
                goal_calc_class = goal_calculators.PlanarMultiLinkJointGoal
                multiproc = True

            self.goal_calculator = goal_calc_class(namelist.tentacle_2D_state_space, shoulder_anchor,
                                                   chain, multiproc=multiproc, init_resp=x_init)
        else:
            raise ValueError("Unrecognized decoder state space!")

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine the target state of the task
        '''
        if isinstance(self.goal_calculator, goal_calculators.PlanarMultiLinkJointGoalCached):
            task_eps = np.inf
        else:
            task_eps = 0.5
        ik_eps = task_eps/10
        data, solution_updated = self.goal_calculator(self.target_location, verbose=False, n_particles=500, eps=ik_eps, n_iter=10, q_start=self.plant.get_intrinsic_coordinates())
        target_state, error = data

        if isinstance(self.goal_calculator, goal_calculators.PlanarMultiLinkJointGoal) and error > task_eps and solution_updated:
            self.goal_calculator.reset()

        return np.array(target_state).reshape(-1,1)

    def _start_wait(self):
        super()._start_wait()
        if self.reset:
            self.decoder.filt.state.mean = self.init_decoder_mean
            self.hdf.sendMsg("reset")

    @classmethod
    def get_desc(cls, params, log_summary):
        duration = round(log_summary['runtime'] / 60, 1)
        return "{}/{} succesful trials in {} min".format(
            log_summary['n_success_trials'], log_summary['n_trials'], duration)

class BMIControlMulti2DWindow(BMIControlMultiMixin, WindowDispl2D, ScreenTargetCapture):
    fps = 20.
    def __init__(self,*args, **kwargs):
        super(BMIControlMulti2DWindow, self).__init__(*args, **kwargs)

    def create_assister(self):
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        if hasattr(self, 'assist_speed'):
            kwargs['assist_speed'] = self.assist_speed
        self.assister = SimpleEndpointAssister(**kwargs)

    def create_goal_calculator(self):
        self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)

    def _start_wait(self):
        self.wait_time = 0.
        super(BMIControlMulti2DWindow, self)._start_wait()

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

class BMIControlMulti(BMIControlMultiMixin, ScreenTargetCapture):
    '''
    Slightly refactored original bmi control task
    '''
    pass

class BMIControlMultiDirectionConstraint(BMIControlMultiMixin, ScreenReachAngle):
    '''
    Adds an additional constraint that the direction of travel must be within a certain angle
    '''
    pass