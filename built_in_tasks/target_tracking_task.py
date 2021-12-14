'''
A generic target tracking task
'''
import numpy as np
import time
import os
import math
import traceback
from collections import OrderedDict

#from features.reward_features import RewardAudio
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
    target_index = 0
    total_distance_error = 0 #Euclidian distance between cursor and target during each trial
    tries = 0 # Helper variable to keep track of the number of failed attempts at a given trial.
    trial_timed_out = False #check if the trial is finished
    sequence_generators = []
    plant_position = []

    reward_time = traits.Float(.5, desc="Length of reward dispensation")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")

    def init(self):
        self.trial_dtype = np.dtype([('trial', 'u4'), ('index', 'u4'), ('target', 'f8', (3,))])
        super().init()
    
    def _parse_next_trial(self):
        '''Check that the generator has the required data'''
        self.gen_indices, self.targs = self.next_trial
        # Update the data sinks with trial information
        self.trial_record['trial'] = self.calc_trial_num()

        self.trial_record['index'] = self.gen_indices
        self.trial_record['target'] = self.targs
        self.sinks.send("trials", self.trial_record)

    def _start_wait(self):
        # Call parent method to draw the next target capture sequence from the generator
        super()._start_wait()

        # number of times this sequence of targets has been attempted
        self.tries = 0
    
        target_index = -1
        # number of targets to be acquired in this trial
        self.chain_length = len(self.targs)

    
    def _start_target(self):
        '''Nothing generic to do.'''
        pass


    def _end_target(self):
        '''Nothing generic to do.'''
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
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[self.target_index])
        return d <= (self.target_radius - self.cursor_radius)

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



class ScreenTargetTracking(TargetTracking, Window):
    """Concrete implementation of Target Tracking task where the target is moving and
    are tracked by holding the cursor within the moving target"""

    limit2d = True
    limit1d = False

    sequence_generators = [
        'rand_target_chain_2D'
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
    #reward_sound = traits.OptionsList("click.wav",desc="File in riglib/audio to play on each reward")
    #myRewardAudio = RewardAudio()

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
        target.move_to_position(self.targs[self.target_index])
        target.show()
        self.sync_event('TARGET_ON', self.gen_indices)
  
    def _start_reward(self):
        self.targets.cue_trial_end_success()
        self.sync_event('REWARD')
        #self.myRewardAudio._start_reward()

    def _end_reward(self):
        super()._end_reward()
        self.sync_event('TRIAL_END')
        # Hide targets
        self.targets.hide()
        self.targets.reset()

    
    ### Generator functions ####
    @staticmethod
    def rand_target_chain_2D(nblocks=100, ntrials=1, boundaries=(-10,10,-10,10)):
        '''
        Generates a sequence of 2D (x and z) target pairs.

        Parameters
        ----------
        ntrials : int
            The number of target chains in the sequence.
        chain_length : int
            The number of targets in each chain
        boundaries: 4 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)

        Returns
        -------
        [ntrials x chain_length x 3] array of target coordinates
        '''
        rng = np.random.default_rng()
        idx = 0
        for i in range(nblocks):
            for j in range(ntrials):
            # Choose a random sequence of points within the boundaries
                pts = rng.uniform(size=(1, 3))*((boundaries[1]-boundaries[0]), 0, (boundaries[3]-boundaries[2]))
                pts = pts+(boundaries[0], 0, boundaries[2])
                yield idx, pts
                idx += 1