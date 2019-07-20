'''
BMI tasks in the new structure, i.e. inheriting from manualcontrolmultitasks
'''
import numpy as np
import time, random

from riglib.experiment import traits, experiment
from features.bmi_task_features import LinearlyDecreasingAssist, LinearlyDecreasingHalfLife

import os
from riglib.bmi import clda, assist, extractor, train, goal_calculators, ppfdecoder
import riglib.bmi
import pdb
import multiprocessing as mp
import pickle
import tables
import re

from riglib.stereo_opengl import ik
import tempfile, pickle, traceback, datetime

from riglib.bmi.bmi import GaussianStateHMM, Decoder, GaussianState, BMISystem, BMILoop
from riglib.bmi.assist import Assister, SSMLFCAssister, FeedbackControllerAssist
from riglib.bmi import feedback_controllers
from riglib.stereo_opengl.window import WindowDispl2D
from riglib.stereo_opengl.primitives import Line


from riglib.bmi.state_space_models import StateSpaceEndptVel2D, StateSpaceNLinkPlanarChain


from . import manualcontrolmultitasks

target_colors = {"blue":(0,0,1,0.5),
"yellow": (1,1,0,0.5),
"hibiscus":(0.859,0.439,0.576,0.5),
"magenta": (1,0,1,0.5),
"purple":(0.608,0.188,1,0.5),
"lightsteelblue":(0.690,0.769,0.901,0.5),
"dodgerblue": (0.118,0.565,1,0.5),
"teal":(0,0.502,0.502,0.5),
"aquamarine":(0.498,1,0.831,0.5),
"olive":(0.420,0.557,0.137,0.5),
"chiffonlemon": (0.933,0.914,0.749,0.5),
"juicyorange": (1,0.502,0,0.5),
"salmon":(1,0.549,0.384,0.5),
"wood": (0.259,0.149,0.071,0.5),
"elephant":(0.409,0.409,0.409,0.5)}


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
class BMIControlMulti(BMILoop, LinearlyDecreasingAssist, manualcontrolmultitasks.ManualControlMulti):
    '''
    Target capture task with cursor position controlled by BMI output.
    Cursor movement can be assisted toward target by setting assist_level > 0.
    '''

    background = (.5,.5,.5,1) # Set the screen background color to grey
    reset = traits.Int(0, desc='reset the decoder state to the starting configuration')

    ordered_traits = ['session_length', 'assist_level', 'assist_level_time', 'reward_time','timeout_time','timeout_penalty_time']
    exclude_parent_traits = ['marker_count', 'marker_num', 'goal_cache_block']

    static_states = [] # states in which the decoder is not run
    hidden_traits = ['arm_hide_rate', 'arm_visible', 'hold_penalty_time', 'rand_start', 'reset', 'target_radius', 'window_size']

    is_bmi_seed = False

    cursor_color_adjust = traits.OptionsList(*list(target_colors.keys()), bmi3d_input_options=list(target_colors.keys()))

    def __init__(self, *args, **kwargs):     
        super(BMIControlMulti, self).__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        sph = self.plant.graphics_models[0]
        sph.color = target_colors[self.cursor_color_adjust]
        sph.radius = self.cursor_radius
        self.plant.cursor_radius = self.cursor_radius   
        self.plant.cursor.radius = self.cursor_radius
        super(BMIControlMulti, self).init(*args, **kwargs)


    def move_effector(self, *args, **kwargs):
        pass

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

    def _end_timeout_penalty(self):
        if self.reset:
            self.decoder.filt.state.mean = self.init_decoder_mean
            self.hdf.sendMsg("reset")

    def move_effector(self):
        pass

    # def _test_enter_target(self, ts):
    #     '''
    #     return true if the distance between center of cursor and target is smaller than the cursor radius
    #     '''
    #     cursor_pos = self.plant.get_endpoint_pos()
    #     d = np.linalg.norm(cursor_pos - self.target_location)
    #     return d <= self.target_radius

class BMIControlMulti2DWindow(BMIControlMulti, WindowDispl2D):
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


class BMIResetting(BMIControlMulti):
    '''
    Task where the virtual plant starts in configuration sampled from a discrete set and resets every trial
    '''
    status = dict(
        wait = dict(start_trial="premove", stop=None),
        premove=dict(premove_complete="target"),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target", trial_restart="premove"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )

    plant_visible = 1
    plant_hide_rate = -1
    premove_time = traits.Float(.1, desc='Time before subject must start doing BMI control')
    # static_states = ['premove'] # states in which the decoder is not run
    add_noise = 0.35
    sequence_generators = BMIControlMulti.sequence_generators + ['outcenter_half_hidden', 'short_long_centerout']

    # def __init__(self, *args, **kwargs):
    #     super(BMIResetting, self).__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        #self.add_dtype('bmi_P', 'f8', (self.decoder.ssm.n_states, self.decoder.ssm.n_states))
        super(BMIResetting, self).init(*args, **kwargs)

    # def move_plant(self, *args, **kwargs):
    #     super(BMIResetting, self).move_plant(*args, **kwargs)
    #     c = self.plant.get_endpoint_pos()
    #     self.plant.set_endpoint_pos(c + self.add_noise*np.array([np.random.rand()-0.5, 0., np.random.rand()-0.5]))

    def _cycle(self, *args, **kwargs):
        #self.task_data['bmi_P'] = self.decoder.filt.state.cov 
        super(BMIResetting, self)._cycle(*args, **kwargs)

    def _while_premove(self):
        self.plant.set_endpoint_pos(self.targs[0])
        self.decoder['q'] = self.plant.get_intrinsic_coordinates()
        # self.decoder.filt.state.mean = self.calc_perturbed_ik(self.targs[0])

    def _start_premove(self):

        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.targets[(self.target_index+1) % 2]
        target.move_to_position(self.targs[self.target_index+1])
        target.cue_trial_start()
        
    def _end_timeout_penalty(self):
        pass

    def _test_premove_complete(self, ts):
        return ts>=self.premove_time

    def _parse_next_trial(self):
        try:
            self.targs, self.plant_visible = self.next_trial        
        except:
            self.targs = self.next_trial

    def _test_hold_complete(self,ts):
        ## Disable origin holds for this task
        if self.target_index == 0:
            return True
        else:
            return ts>=self.hold_time

    def _test_trial_incomplete(self, ts):
        return (self.target_index<self.chain_length-1) and (self.target_index != -1) and (self.tries<self.max_attempts)

    def _test_trial_restart(self, ts):
        return (self.target_index==-1) and (self.tries<self.max_attempts)

    @staticmethod
    def outcenter_half_hidden(nblocks=100, ntargets=4, distance=8, startangle=45):
        startangle = np.deg2rad(startangle)
        target_angles = np.arange(startangle, startangle+2*np.pi, 2*np.pi/ntargets)
        origins = distance * np.vstack([np.cos(target_angles), 
                                        np.zeros_like(target_angles),
                                        np.sin(target_angles)]).T
        terminus = np.zeros(3)
        trial_target_sequences = [np.vstack([origin, terminus]) for origin in origins]
        visibility = [True, False]
        from riglib.experiment.generate import block_random
        seq = block_random(trial_target_sequences, visibility, nblocks=nblocks)
        return seq
    
    @staticmethod
    def short_long_centerout(nblocks=100, ntargets=4, distance2=(8, 12)):
        theta = []
        dist = []
        for i in range(nblocks):
            for j in range(2):
                if j==0:
                    temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
                    tempdist = np.zeros((ntargets, )) + distance2[j]
                else:
                    temp = np.hstack((temp, np.arange(0, 2*np.pi, 2*np.pi/ntargets)))
                    tempdist = np.hstack((tempdist, np.zeros((ntargets, ))+distance2[j]))
            
            ix = np.random.permutation(ntargets*2)
            theta = theta + [temp[ix]]
            dist = dist + list(tempdist[ix])
        theta = np.hstack(theta)
        distance = np.hstack(dist)
        
        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs


class BaselineControl(BMIControlMulti):
    background = (0.0, 0.0, 0.0, 1) # Set background to black to make it appear to subject like the task is not running

    def show_object(self, obj, show=False):
        '''
        Show or hide an object
        '''
        obj.detach()

    def init(self, *args, **kwargs):
        super(BaselineControl, self).init(*args, **kwargs)

    def _cycle(self, *args, **kwargs):
        for model in self.plant.graphics_models:
            model.detach()
        super(BaselineControl, self)._cycle(*args, **kwargs)

    def _start_wait(self, *args, **kwargs):
        for model in self.plant.graphics_models:
            model.detach()
        super(BaselineControl, self)._start_wait(*args, **kwargs)


#########################
######## Simulation tasks
#########################
from features.simulation_features import SimKalmanEnc, SimKFDecoderSup, SimCosineTunedEnc
from riglib.bmi.feedback_controllers import LQRController
class SimBMIControlMulti(SimCosineTunedEnc, SimKFDecoderSup, BMIControlMulti):
    win_res = (250, 140)
    sequence_generators = ['sim_target_seq_generator_multi']
    def __init__(self, *args, **kwargs):
        from riglib.bmi.state_space_models import StateSpaceEndptVel2D
        ssm = StateSpaceEndptVel2D()

        A, B, W = ssm.get_ssm_matrices()
        Q = np.mat(np.diag([1., 1, 1, 0, 0, 0, 0]))
        R = 10000*np.mat(np.diag([1., 1., 1.]))
        self.fb_ctrl = LQRController(A, B, Q, R)

        self.ssm = ssm

        super(SimBMIControlMulti, self).__init__(*args, **kwargs)

    @staticmethod
    def sim_target_seq_generator_multi(n_targs, n_trials):
        '''
        Simulated generator for simulations of the BMIControlMulti and CLDAControlMulti tasks
        '''
        center = np.zeros(2)
        pi = np.pi
        targets = 8*np.vstack([[np.cos(pi/4*k), np.sin(pi/4*k)] for k in range(8)])

        target_inds = np.random.randint(0, n_targs, n_trials)
        target_inds[0:n_targs] = np.arange(min(n_targs, n_trials))
        for k in range(n_trials):
            targ = targets[target_inds[k], :]
            yield np.array([[center[0], 0, center[1]],
                            [targ[0], 0, targ[1]]])        
