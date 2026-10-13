'''
BMI tasks in the new structure, i.e. inheriting from manualcontrolmultitasks
'''
import numpy as np
import pickle

from built_in_tasks.target_tracking_task import ScreenTargetTracking
from riglib.experiment import traits

from riglib.bmi import goal_calculators, ppfdecoder, feedback_controllers
from riglib.bmi.bmi import BMILoop
from riglib.bmi.assist import Assister, FeedbackControllerAssist
from riglib.bmi.state_space_models import StateSpaceEndptVel2D, StateSpaceNLinkPlanarChain
from riglib.experiment.experiment import control_decorator

from riglib.stereo_opengl.window import WindowDispl2D
from .target_capture_task import ScreenTargetCapture
from .target_capture_task_xt import ScreenReachAngle
from features.bmi_task_features import LinearlyDecreasingAssist
from .target_graphics import target_colors
from .target_tracking_task import ScreenTargetTracking

from .target_capture_task_eye import EyeConstrainedTargetCapture

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
        OFCEndpointAssister instanBMIControlMultiMixince
        '''
        F_dict = pickle.load(open('/storage/assist_params/assist_20levels_ppf.pkl'))
        B = np.asmatrix(np.vstack([np.zeros([3,3]), np.eye(3)*1000*1./decoding_rate, np.zeros(3)]))
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
        F = np.asmatrix(self.fb_ctrl.F_dict[assist_level_idx])
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
        self.assist_noise = kwargs.pop('assist_noise', 0.)

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
            Bu = self.endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen, speed, target_radius, self.assist_noise)
            assist_weight = assist_level

        # return Bu, assist_weight
        return dict(x_assist=Bu, assist_level=assist_weight)

    @staticmethod
    def endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen=0.1, speed=0.5, target_radius=2., assist_noise=0.):
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
        assist_noise: float
            Noise added to the assist speed to vary the timing of the trajectories

        Returns
        -------
        x_assist : np.ndarray of shape (7, 1)
            Control vector to add onto the state vector to assist control.
        '''
        diff_vec = target_pos - cursor_pos
        dist_to_target = np.linalg.norm(diff_vec)
        dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)

        if dist_to_target > target_radius:
            assist_cursor_pos = cursor_pos + np.random.uniform(1-assist_noise,1+assist_noise)*speed*dir_to_target
        else:
            assist_cursor_pos = cursor_pos + np.random.uniform(1-assist_noise,1+assist_noise)*speed*diff_vec/2

        assist_cursor_vel = (assist_cursor_pos-cursor_pos)/decoder_binlen
        x_assist = np.hstack([assist_cursor_pos, assist_cursor_vel, 1])
        x_assist = np.asmatrix(x_assist.reshape(-1,1))
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
        A = np.asmatrix([[1., 0, 0, dt, 0, 0, 0],
                    [0., 1, 0, 0,  dt, 0, 0],
                    [0., 0, 1, 0, 0, dt, 0],
                    [0., 0, 0, 0, 0,  0, 0],
                    [0., 0, 0, 0, 0,  0, 0],
                    [0., 0, 0, 0, 0,  0, 0],
                    [0., 0, 0, 0, 0,  0, 1]])

        I = np.asmatrix(np.eye(3))
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
    reset = traits.Int(0, desc='reset the decoder state to the starting configuration. 1 for always, 2 for only on timeout')
    auto_reset_time = traits.Float(0, desc='automatically reset the decoder state every N seconds. 0 for no auto-reset')
    assist_speed = traits.Float(2., desc="speed of assister in cm/s")
    assist_noise = traits.Float(0., desc="noise added to cursor speed in cm/s")
    cursor_color = traits.OptionsList("orange", *target_colors, desc='Color of cursor endpoint', bmi3d_input_options=list(target_colors.keys()))
    save_zscore = traits.Bool(False, desc="save a decoder zscored from this task")
    dist_tol_reset = traits.Float(0.001, desc="distance from the edge of the workspace to trigger auto-reset")

    static_states = ['reward'] # states in which the decoder is not run

    ordered_traits = ['session_length', 'assist_level', 'assist_level_time', 'reward_time','timeout_time','timeout_penalty_time']
    exclude_parent_traits = ['marker_count', 'marker_num', 'goal_cache_block']
    hidden_traits = ['arm_hide_rate', 'arm_visible', 'hold_penalty_time', 'rand_start', 'reset', 'target_radius', 'window_size']

    is_bmi_seed = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.auto_reset_starter = None # time when the auto-reset timer started

    def _cycle(self):
        super()._cycle()
        if self.auto_reset_time <= 0:
            return 

        rel_time = self.get_time()
        # function to check if cursor is at edge of the workspace. Start timer & reset if it has been there longer than reset time 
        # does not move cursor to bounds if far exceeding boundary, just counts the time 
        if self._cursor_at_bounds():
            if self.auto_reset_starter is None:
                self.auto_reset_starter = rel_time #start timer using current time 
            elif (rel_time - self.auto_reset_starter) > self.auto_reset_time: #reset if cursor has been at edge for long enough 
                self.hdf.sendMsg("auto-reset")
                self.reset_cursor()
                self.auto_reset_starter = None
        else:
            self.auto_reset_starter = None
     

    def _cursor_at_bounds(self):  
        pos = np.asarray(self.plant.get_endpoint_pos()).ravel()
        bounds = np.asarray(self.cursor_bounds)      
        lo, hi = bounds[0::2], bounds[1::2] # pair up bounds for each diimension 
        active = hi > lo # ignore an axis that has no range 
        at_edge = (np.abs(pos - lo) < self.dist_tol_reset) | (np.abs(pos - hi) < self.dist_tol_reset)
        return np.any(at_edge & active)
        
    def move_effector(self, *args, **kwargs):
        pass
    
    def create_assister(self):
        # Create the appropriate type of assister object
        start_level, end_level = self.assist_level
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        kwargs['assist_speed'] = self.assist_speed
        kwargs['assist_noise'] = self.assist_noise

        if isinstance(self.decoder.ssm, StateSpaceEndptVel2D) and isinstance(self.decoder, ppfdecoder.PPFDecoder):
            self.assister = OFCEndpointAssister()
        elif isinstance(self.decoder.ssm, StateSpaceEndptVel2D):
            self.assister = SimpleEndpointAssister(**kwargs)
        ## elif (self.decoder.ssm == namelist.tentacle_2D_state_space) or (self.decoder.ssm == namelist.joint_2D_state_space):
        ##     # kin_chain = self.plant.kin_chain
        ##     # A, B, W = self.decoder.ssm.get_ssm_matrices(update_rate=self.decoder.binlen)
        ##     # Q = np.asmatrix(np.diag(np.hstack([kin_chain.link_lengths, np.zeros_like(kin_chain.link_lengths), 0])))
        ##     # R = 10000*np.asmatrix(np.eye(B.shape[1]))

        ##     # fb_ctrl = LQRController(A, B, Q, R)
        ##     # self.assister = FeedbackControllerAssist(fb_ctrl, style='additive')
        ##     self.assister = TentacleAssist(ssm=self.decoder.ssm, kin_chain=self.plant.kin_chain, update_rate=self.decoder.binlen)
        else:
            raise NotImplementedError("Cannot assist for this type of statespace: %r" % self.decoder.ssm)

        print('Assister: ', self.assister)

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
            x_init = np.asmatrix(x_init).reshape(-1, 1)

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

    
    def move_effector(self, pos_offset=[0,0,0], vel_offset=[0,0,0]):
        pass
    
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

    def _end_targ_transition(self):
        super()._end_targ_transition()
        if self.reset == 1 and ((self.target_index == self.chain_length - 1) or (self.target_index == -1)):

            # Reset on any target transition away from the last target
            self.decoder.filt.state.mean = self.init_decoder_mean.copy()
            self.hdf.sendMsg("reset")

    def _end_timeout_penalty(self):
        super()._end_timeout_penalty()
        if self.reset == 2:

            # Reset on timeout
            self.decoder.filt.state.mean = self.init_decoder_mean.copy()
            self.hdf.sendMsg("reset")

    def _start_None(self):
        super()._start_None()

        # Optionally save a new decoder zscored from this task
        if (not self.save_zscore) or (self.saveid is None):
            return
        #Check if null decoder here
        if hasattr(self.decoder, 'is_null_decoder') and self.decoder.is_null_decoder:
            return
        
        if not (np.all(self.decoder.mFR == 0) and np.all(self.decoder.sdFR == 1)):
            filename = self.decoder.save()

            from db.tracker import dbq
            suffix = f"saved_from_{self.saveid}"
            dbq.save_bmi(suffix, self.saveid, filename)
            return

        # This is linear mapping specific - needs updating
        self.decoder.filt.fix_norm_attr()
        self.decoder.filt._update_scale_attr()
        mFR = self.decoder.filt.attr['offset'].copy()
        sdFR = self.decoder.filt.attr['scale'].copy()
        n_units = self.decoder.filt.n_units
        self.decoder.filt.update_norm_attr(offset=np.zeros(n_units), scale=np.ones(n_units))

        # The rest should work with any decoder
        self.decoder.init_zscore(mFR, sdFR)
        
        filename = self.decoder.save()

        from db.tracker import dbq
        suffix = f"zscored_online_in_{self.saveid}"
        dbq.save_bmi(suffix, self.saveid, filename)


    @classmethod
    def get_desc(cls, params, log_summary):
        duration = round(log_summary['runtime'] / 60, 1)
        return "{}/{} succesful trials in {} min".format(
            log_summary['n_success_trials'], log_summary['n_trials'], duration)

    @control_decorator
    def reset_cursor(self):
        self.decoder.filt.state.mean = self.init_decoder_mean.copy()
        self.hdf.sendMsg("reset")
        
    # @control_decorator
    # def update_zscore(self):
    #     # This is a temporary solution - LRS Jan 2024
    #     # Only works if the decoder currently has mFR == 0 and sdFR == 1
    #     # Assumes you're using a lindecoder with a reasonably large buffer (>2 minutes)
    #     self.decoder.filt.fix_norm_attr() # Should be already!
    #     self.decoder.filt._update_scale_attr()
    #     print(f"updated zscore")
    #     self.hdf.sendMsg(f"updated zscore")
    #     mFR = self.decoder.filt.attr['offset'].copy()
    #     sdFR = self.decoder.filt.attr['scale'].copy()
    #     self.decoder.init_zscore(mFR, sdFR)
    #     self.hdf.sendAttr("task", "session_mFR", mFR)
    #     self.hdf.sendAttr("task", "session_sdFR", sdFR)
    #     n_units = self.decoder.filt.n_units
    #     self.decoder.filt.update_norm_attr(offset=np.zeros(n_units), scale=np.ones(n_units))

    @control_decorator
    def toggle_clda(self):
        self.learn_flag = not self.learn_flag
        self.hdf.sendMsg(f"clda = {self.learn_flag}")
        print(f"clda = {self.learn_flag}")

    

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

class BMIControlMultiEyeConstrained(BMIControlMultiMixin, EyeConstrainedTargetCapture):
    '''
    Slightly refactored original bmi control task
    '''
    pass

class BMIControlMultiDirectionConstraint(BMIControlMultiMixin, ScreenReachAngle):
    '''
    Adds an additional constraint that the direction of travel must be within a certain angle
    '''
    pass


class BMIControlMulti_ScreenTargetTracking(BMIControlMultiMixin, ScreenTargetTracking):
    '''
    BMI control in the tracking task. 
    '''
    
    def move_effector(self, pos_offset=[0,0,0], vel_offset=[0,0,0]):
        '''Runs after BMILoop.move_plant() has set the cursor to the decoded position.'''
        pos_offset = np.asarray(pos_offset, dtype=float)
        if not np.any(pos_offset):
            return
        decoded_pos = np.asarray(self.plant.get_endpoint_pos()).ravel()
        new_pos = decoded_pos + pos_offset

        bounds = np.asarray(self.cursor_bounds)       # (min0, max0, min1, max1, min2, max2)
        new_pos = np.clip(new_pos, bounds[0::2], bounds[1::2])
        self.plant.set_endpoint_pos(new_pos)
