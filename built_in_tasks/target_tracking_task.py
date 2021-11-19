'''
A generic target tracking task
'''
import numpy as np
import time
import os
import math
import traceback
from collections import OrderedDict

from riglib.experiment import traits, Sequence, FSMTable, StateTransitions
from riglib.stereo_opengl import ik
from riglib import plants

from riglib.stereo_opengl.window import Window
from .target_graphics import *

## Plants
# List of possible "plants" that a subject could control either during manual or brain control
cursor = plants.CursorPlant()
shoulder_anchor = np.array([2., 0., -15])
chain_kwargs = dict(link_radii=.6, joint_radii=0.6, joint_colors=(181/256., 116/256., 96/256., 1), link_colors=(181/256., 116/256., 96/256., 1))
chain_20_20_endpt = plants.EndptControlled2LArm(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)
chain_20_20 = plants.RobotArmGen2D(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)

plantlist = dict(
    cursor=cursor,
    chain_20_20=chain_20_20,
    chain_20_20_endpt=chain_20_20_endpt)

class TargetTracking(Sequence):
    '''
    This is a generic cued target tracking skeleton, to form as a common ancestor to the most
    common type of motor control tracking tasks.
    '''
    status = dict(
        wait = dict(start_trial="target"),
        target = dict(success="reward", timeout="wait"),
        reward = dict(reward_end="wait", stoppable=False, end_state=True)
    )

    # initial state
    state = "wait"
    frame_index = 0 #index in the frame in a trial
    total_distance_error = 0 #Euclidian distance between cursor and target during each trial
    tries = 0 # Helper variable to keep track of the number of failed attempts at a given trial.
    trial_timed_out = False #check if the trial is finished
    sequence_generators = []
    plant_position = []
    disterbance_trial = False
    disterbance_position = None

    reward_time = traits.Float(.5, desc="Length of reward dispensation")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")

    def init(self):
        self.trial_dtype = np.dtype([('trial', 'u4'), ('index', 'u4'), ('target', 'f8', (300,3)), ('disterbance_path', 'f8', (300,)), ('is_disterbance', 'u4')])
        super().init()
    
    def _parse_next_trial(self):
        '''Check that the generator has the required data'''
        self.gen_indices, self.targs, self.disterbance_trial, self.disterbance_path = self.next_trial
        self.targs = np.squeeze(self.targs,axis=0)
        self.disterbance_path = np.squeeze(self.disterbance_path)

        # Update the data sinks with trial information
        self.trial_record['trial'] = self.calc_trial_num()
        #import pdb; pdb.set_trace()

        self.trial_record['index'] = self.gen_indices
        self.trial_record['target'] = self.targs
        self.trial_record['disterbance_path'] = self.disterbance_path
        self.trial_record['is_disterbance'] = self.disterbance_trial
        self.sinks.send("trials", self.trial_record)

    def _start_wait(self):
        # Call parent method to draw the next target capture sequence from the generator
        super()._start_wait()

        # number of times this sequence of targets has been attempted
        self.tries = 0

        # number of targets to be acquired in this trial
        self.chain_length = len(self.targs)

    
    def _start_target(self):
        self.frame_index = 0
        self.total_distance_error = 0
        self.plant_position.append(self._get_manual_position()[0])

    def _while_target(self):
        self.total_distance_error += self.test_in_target() #Calculate and sum distance between center of cursor and current target position
        
        #Add Disterbance TODO

        self.plant.set_endpoint_pos(np.array([0,0,0]))
        
        self.plant_position.append(self._get_manual_position()[0])
       
        self.disterbance_position = self.add_disterbance(self.plant_position[-1], self.plant_position[-1]-self.plant_position[-2], self.disterbance_path[self.frame_index])

        #Move Target to next frame so it appears to be moving
        target = self.targets
        target.move_to_position(self.targs[self.frame_index])
        target.show()
        self.sync_event('MOVE_TARGET', self.targs[self.frame_index])
        self.frame_index +=1

        #Check if the trial is over and there are no more target frames to display
        if np.shape(self.targs)[0] <= self.frame_index:
            self.trial_timed_out = True     

    def _end_target(self):
        self.trial_timed_out = False
        pass

    def _start_reward(self):
        '''Nothing generic to do.'''
        pass

    def _while_reward(self):
        '''Nothing generic to do.'''
        pass

    def _end_reward(self):
        '''Nothing generic to do.'''
        pass

    ################## State transition test functions ##################
    def _test_start_trial(self, time_in_state):
        '''Start next trial automatically. You may want this to instead be
            - a random delay
            - require some initiation action
        '''
        return True

    def _test_success(self, time_in_state):
        '''
        Test if the current state is successful and should transition to reward state.
        In order to be successful it needs to be the end of the trial and the total 
        distance error must be less than 2 cm on average of the whole trial.
        This means the center of the cursor was on average within 2cm of the center of the target.
        '''
        return self.trial_timed_out and self.total_distance_error/self.frame_index < 2

    def _test_timeout(self, time_in_state):
        '''
        This test if the trial has finshed unsuccessfully and should transition to the wait state
        _test_success() is called before this test function and so if the trial is over, but successful
        it would transition to reward otherwise it will transition to the wait state here.
        '''
        return  self.trial_timed_out or self.pause

    def _test_reward_end(self, time_in_state):
        '''
        Test the reward state has ended.
        '''
        return time_in_state > self.reward_time

    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super().update_report_stats()
        self.reportstats['Trial #'] = self.calc_trial_num()
        self.reportstats['Reward/min'] = np.round(self.calc_events_per_min('reward', 120.), decimals=2)
    
    def test_in_target(self):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[self.frame_index])
        if d <= (self.target_radius - self.cursor_radius):
            self.targets.cue_trial_end_success()
        else:
            self.targets.reset()
        return d
    
    def add_disterbance(self, current_position, current_velocity, disterbance):
        return  current_position + current_velocity + disterbance


class ScreenTargetTracking(TargetTracking, Window):
    """Concrete implementation of Target Tracking task where the target is moving and
    are tracked by holding the cursor within the moving target"""

    limit2d = True
    limit1d = True

    sequence_generators = [
        'tracking_target_chain_1D', 'tracking_target_chain_2D'
    ]

    hidden_traits = ['cursor_color', 'target_color', 'cursor_bounds', 'cursor_radius', 'plant_hide_rate', 'starting_pos']

    is_bmi_seed = True

    # Runtime settable traits
    target_radius = traits.Float(1.5, desc="Radius of targets in cm")
    target_color = traits.OptionsList("yellow", *target_colors, desc="Color of the target", bmi3d_input_options=list(target_colors.keys()))
    plant_hide_rate = traits.Float(0.0, desc='If the plant is visible, specifies a percentage of trials where it will be hidden')
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))
    plant_visible = traits.Bool(True, desc='Specifies whether entire plant is displayed or just endpoint')
    cursor_radius = traits.Float(.5, desc='Radius of cursor in cm')
    cursor_color = traits.OptionsList("pink", *target_colors, desc='Color of cursor endpoint', bmi3d_input_options=list(target_colors.keys()))
    cursor_bounds = traits.Tuple((-10., 10., 0., 0., -10., 10.), desc='(x min, x max, y min, y max, z min, z max)')
    starting_pos = traits.Tuple((5., 0., 5.), desc='Where to initialize the cursor') 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the plant
        if not hasattr(self, 'plant'):
            self.plant = plantlist[self.plant_type]
        self.plant.set_bounds(np.array(self.cursor_bounds))
        self.plant.set_color(target_colors[self.cursor_color])
        self.plant.set_cursor_radius(self.cursor_radius)
        self.plant_vis_prev = True
        self.cursor_vis_prev = True

        # Add graphics models for the plant and targets to the window
        if hasattr(self.plant, 'graphics_models'):
            for model in self.plant.graphics_models:
                self.add_model(model)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            # Need two targets to have the ability for delayed holds
            self.targets = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
        

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)

    def init(self):
        self.add_dtype('trial', 'u4', (1,))
        self.add_dtype('plant_visible', '?', (1,))
        super().init()
        self.plant.set_endpoint_pos(np.array(self.starting_pos))

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen
        '''
        if self.disterbance_trial:
            self.move_effector(self.disterbance_position)
        else:
            self.move_effector()
        
        ## Run graphics commands to show/hide the plant if the visibility has changed
        self.update_plant_visibility()
        self.task_data['plant_visible'] = self.plant_visible

        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        # Update the trial index
        self.task_data['trial'] = self.calc_trial_num()

        super()._cycle()

    def move_effector(self):
        '''Move the end effector, if a robot or similar is being controlled'''
        pass

    def run(self):
        '''
        See experiment.Experiment.run for documentation.
        '''
        # Fire up the plant. For virtual/simulation plants, this does little/nothing.
        self.plant.start()
        
        # Include some cleanup in case the parent class has errors
        try:
            super().run()
        finally:
            self.plant.stop()

    ##### HELPER AND UPDATE FUNCTIONS ####
    def update_plant_visibility(self):
        ''' Update plant visibility'''
        if self.plant_visible != self.plant_vis_prev:
            self.plant_vis_prev = self.plant_visible
            self.plant.set_visibility(self.plant_visible)

    #### STATE FUNCTIONS ####
    def _start_wait(self):
        super()._start_wait()
        if self.calc_trial_num() == 0:
            # Instantiate the targets here so they don't show up in any states that might come before "wait"
            for model in self.targets.graphics_models:
                self.add_model(model)
                self.targets.hide()

    def _start_target(self):
        super()._start_target()
        # Show target if it is hidden (this is the first target, or previous state was a penalty)
        target = self.targets  
        target.move_to_position(self.targs[self.frame_index])
        target.show()
        self.sync_event('TARGET_ON', self.gen_indices)
  
    def _start_reward(self):
        self.targets.cue_trial_end_success()
        self.sync_event('REWARD')
    
    def _end_reward(self):
        super()._end_reward()
        self.sync_event('TRIAL_END')
        # Hide targets
        self.targets.hide()
        self.targets.reset()

    @staticmethod
    def calc_sum_of_sines(times, frequencies, amplitudes, phase_shifts):

        t = times
        t = np.asarray(t).copy(); t.shape = (t.size,1)

        f = frequencies
        f = f.copy(); f.shape = (1,f.size)

        a = amplitudes
        a = a.copy(); a.shape = (1,a.size)

        p = phase_shifts
        p = p.copy(); p.shape = (1,p.size)

        assert f.shape == a.shape == p.shape,"Shape of frequencies, amplitudes, and phase shifts must match"

        o = np.ones(t.shape)
        _ = np.sum(np.dot(o,a) * np.sin(2*np.pi*(np.dot(t,f) + np.dot(o,p))),axis=1)

        return _

    @staticmethod
    def calc_sum_of_sines_ramp(times, ramp, frequencies, amplitudes, phase_shifts):

            t = times
            t = np.asarray(t).copy(); t.shape = (t.size,1)

            r = ramp

            _ = ScreenTargetTracking.calc_sum_of_sines(t, frequencies, amplitudes, phase_shifts)

            if r > 0:
                _ *= ((t*(t <= r)/r + (t > r)).flatten())**2

            return _

    @staticmethod
    def generate_trajectory(primes):
        hz = 60 # Hz -- sampling rate
        dt = 1/hz # sec -- sampling period

        T0 = 20 # sec -- base period
        w0 = 1/T0 # Hz -- base frequency

        P = 2 # number of periods in signal
        T = P*T0 # sec -- signal duration
        r = 5 # "ramp" duration (see sum_of_sines_ramp)
        dw = 1/T # Hz -- frequency resolution
        W = dw*T/dt/2 # Hz -- signal bandwidth

        #p = np.array([2,3,5,7,11,13,17,19]) # primes
        #p_y = np.array([]) # 

        f = primes*w0 # stimulated frequencies
        a = 1/(1+np.arange(f.size)) # amplitude
        o = 2*np.pi*np.random.rand(primes.size) # phase offset

        t = np.arange(0,T,dt) # time samples
        w = np.arange(0,W,dw) # frequency samples

        N = t.size # = T/dt -- number of samples

        trajectory = ScreenTargetTracking.calc_sum_of_sines_ramp(t, r, f, a, o/(2*np.pi))
        normalized_trajectory = trajectory/np.sum(a)
        return normalized_trajectory

    ### Generator functions ####
    @staticmethod
    def tracking_target_chain_1D(nblocks=1, ntrials=2, time_length = 5, boundaries=(-10,10,-10,10)):
        '''
        Generates a sequence of 1D (z axis (vertical axis)) target trajectories

        Parameters
        ----------
        nblocks : int
            The number of tracking trials in the sequence.
        ntrials : int
            The number trials in a block
        time_length : int
            The length of one target tracking trial in seconds 
        boundaries: 4 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)

        Returns
        -------
        [nblocks*ntrials x 1] array of tuples containing trial indices and [time_length*60 x 3] target coordinates
        '''
        idx = 0

        disterbance_primes_freq = np.array([2, 5, 11, 17, 23, 31, 41])
        disterbance_trials = np.random.randint(0,nblocks*ntrials,round(nblocks*ntrials*0.5))
        y_primes_freq = np.array([3, 7, 13, 19, 29, 37, 43])
        for i in range(nblocks):
            for j in range(ntrials):
                disterbance = False
                disterbance_path = np.zeros((time_length*60,1))
                trajectory = np.zeros((time_length*60,3))
                sum_of_sins_path = ScreenTargetTracking.generate_trajectory(y_primes_freq)
                rand_start_index = np.random.randint(0,np.shape(sum_of_sins_path)[0]-(time_length*60))
                pts = []
                trajectory[:,2] = 4*sum_of_sins_path[rand_start_index:rand_start_index+time_length*60]
                if idx == disterbance_trials:
                    disterbance_path = 4*ScreenTargetTracking.generate_trajectory(disterbance_primes_freq)[rand_start_index:rand_start_index+time_length*60]
                    disterbance = True
                pts.append(trajectory)
                yield idx, pts, disterbance, disterbance_path
                idx += 1
    
    @staticmethod
    def tracking_target_chain_2D(nblocks=1, ntrials=2, time_length = 5, boundaries=(-10,10,-10,10)):
        '''
        Generates a sequence of 2D (x and z axis) target trajectories

        Parameters
        ----------
        nblocks : int
            The number of tracking trials in the sequence.
        ntrials : int
            The number trials in a block
        time_length : int
        boundaries: 4 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)

        Returns
        -------
        [nblocks*ntrials x 1] array of tuples containing trial indices and [time_length*60 x 3] target coordinates
        '''
        idx = 0
        x_primes_freq = np.array([2, 5, 11, 17, 23, 31, 41])
        y_primes_freq = np.array([3, 7, 13, 19, 29, 37, 43])
        for i in range(nblocks):
            for j in range(ntrials):
                trajectory = np.zeros((time_length*60,3))
                sum_of_sins_pathx = ScreenTargetTracking.generate_trajectory(x_primes_freq)
                sum_of_sins_pathy = ScreenTargetTracking.generate_trajectory(y_primes_freq)
                rand_start_index = np.random.randint(0,np.shape(sum_of_sins_pathy)[0]-(time_length*60))
                pts = []
                trajectory[:,0] = 4*sum_of_sins_pathx[rand_start_index:rand_start_index+time_length*60]
                trajectory[:,2] = 4*sum_of_sins_pathy[rand_start_index:rand_start_index+time_length*60]
                pts.append(trajectory)
                yield idx, pts
                idx += 1