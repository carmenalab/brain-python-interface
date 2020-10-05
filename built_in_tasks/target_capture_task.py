'''
A generic target capture task
'''
import numpy as np
import time
import os
import math
import traceback
from collections import OrderedDict

from riglib.experiment import traits, Sequence, FSMTable, StateTransitions
from riglib.stereo_opengl import ik

from .target_graphics import *
from .plantlist import plantlist


####### CONSTANTS
sec_per_min = 60.0
RED = (1,0,0,.5)
GREEN = (0,1,0,0.5)
GOLD = (1., 0.843, 0., 0.5)
mm_per_cm = 1./10



class TargetCapture(Sequence):
    '''
    This is a generic cued target capture skeleton, to form as a common ancestor to the most
    common type of motor control task.
    '''
    status = FSMTable(
        wait = StateTransitions(start_trial="target"),
        target = StateTransitions(enter_target="hold", timeout="timeout_penalty"),
        hold = StateTransitions(leave_early="hold_penalty", hold_complete="targ_transition"),
        targ_transition = StateTransitions(trial_complete="reward", trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = StateTransitions(timeout_penalty_end="targ_transition", end_state=True),
        hold_penalty = StateTransitions(hold_penalty_end="targ_transition", end_state=True),
        reward = StateTransitions(reward_end="wait", stoppable=False, end_state=True)
    )

    trial_end_states = ['reward', 'timeout_penalty', 'hold_penalty']


    # initial state
    state = "wait"

    target_index = -1 # Helper variable to keep track of which target to display within a trial
    tries = 0 # Helper variable to keep track of the number of failed attempts at a given trial.

    sequence_generators = []

    reward_time = traits.Float(.5, desc="Length of reward dispensation")
    hold_time = traits.Float(.2, desc="Length of hold required at targets")
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")
    timeout_penalty_time = traits.Float(1, desc="Length of penalty time for timeout error")
    max_attempts = traits.Int(10, desc='The number of attempts at a target before\
        skipping to the next one')

    def _start_wait(self):
        # Call parent method to draw the next target capture sequence from the generator
        super()._start_wait()

        # number of times this sequence of targets has been attempted
        self.tries = 0

        # index of current target presented to subject
        self.target_index = -1

        # number of targets to be acquired in this trial
        self.chain_length = len(self.targs)

    def _parse_next_trial(self):
        '''Check that the generator has the required data'''
        self.targs = self.next_trial

        # TODO error checking

    def _start_target(self):
        self.target_index += 1
        self.target_location = self.targs[self.target_index]

    def _end_target(self):
        '''Nothing generic to do.'''
        pass

    def _start_hold(self):
        '''Nothing generic to do.'''
        pass

    def _while_hold(self):
        '''Nothing generic to do.'''
        pass

    def _end_hold(self):
        '''Nothing generic to do.'''
        pass

    def _start_targ_transition(self):
        '''Nothing generic to do. Child class might show/hide targets'''
        pass

    def _while_targ_transition(self):
        '''Nothing generic to do.'''
        pass

    def _end_targ_transition(self):
        '''Nothing generic to do.'''
        pass

    def _start_timeout_penalty(self):
        self.tries += 1
        self.target_index = -1

    def _while_timeout_penalty(self):
        '''Nothing generic to do.'''
        pass

    def _end_timeout_penalty(self):
        '''Nothing generic to do.'''
        pass

    def _start_hold_penalty(self):
        self.tries += 1
        self.target_index = -1

    def _while_hold_penalty(self):
        '''Nothing generic to do.'''
        pass

    def _end_hold_penalty(self):
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

    def _test_timeout(self, time_in_state):
        return time_in_state > self.timeout_time

    def _test_hold_complete(self, time_in_state):
        '''
        Test whether the target is held long enough to declare the
        trial a success

        Possible options
            - Target held for the minimum requred time (implemented here)
            - Sensorized object moved by a certain amount
            - Sensorized object moved to the required location
            - Manually triggered by experimenter
        '''
        return time_in_state > self.hold_time

    def _test_trial_complete(self, time_in_state):
        '''Test whether all targets in sequence have been acquired'''
        return self.target_index == self.chain_length-1

    def _test_trial_abort(self, time_in_state):
        '''Test whether the target capture sequence should just be skipped due to too many failures'''
        return (not self._test_trial_complete(time_in_state)) and (self.tries==self.max_attempts)

    def _test_trial_incomplete(self, time_in_state):
        '''Test whether the target capture sequence needs to be restarted'''
        return (not self._test_trial_complete(time_in_state)) and (self.tries<self.max_attempts)

    def _test_timeout_penalty_end(self, time_in_state):
        return time_in_state > self.timeout_penalty_time

    def _test_hold_penalty_end(self, time_in_state):
        return time_in_state > self.hold_penalty_time

    def _test_reward_end(self, time_in_state):
        return time_in_state > self.reward_time

    def _test_enter_target(self, time_in_state):
        '''This function is task-specific and not much can be done generically'''
        return False

    def _test_leave_early(self, time_in_state):
        '''This function is task-specific and not much can be done generically'''
        return False

    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super().update_report_stats()
        self.reportstats['Trial #'] = self.calc_trial_num()
        self.reportstats['Reward/min'] = np.round(self.calc_events_per_min('reward', 120.), decimals=2)

    @classmethod
    def get_desc(cls, params, report):
        '''Used by the database infrasturcture to generate summary stats on this task'''
        duration = report[-1][-1] - report[0][-1]
        reward_count = 0
        for item in report:
            if item[0] == "reward":
                reward_count += 1
        return "{} rewarded trials in {} min".format(reward_count, duration)


class ScreenTargetCapture(TargetCapture, Window):
    """Concrete implementation of TargetCapture task where targets
    are acquired by "holding" a cursor in an on-screen target"""
    background = (0,0,0,1)
    cursor_color = (.5,0,.5,1)

    plant_type = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=list(plantlist.keys()))

    starting_pos = (5, 0, 5)

    target_color = (1,0,0,.5)

    cursor_visible = False # Determines when to hide the cursor.
    no_data_count = 0 # Counter for number of missing data frames in a row
    scale_factor = 3.0 #scale factor for converting hand movement to screen movement (1cm hand movement = 3.5cm cursor movement)

    limit2d = 1

    sequence_generators = ['centerout_2D_discrete']

    is_bmi_seed = True
    _target_color = RED


    # Runtime settable traits
    reward_time = traits.Float(.5, desc="Length of juice reward")
    target_radius = traits.Float(2, desc="Radius of targets in cm")

    hold_time = traits.Float(.2, desc="Length of hold required at targets")
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")
    timeout_penalty_time = traits.Float(1, desc="Length of penalty time for timeout error")
    max_attempts = traits.Int(10, desc='The number of attempts at a target before\
        skipping to the next one')

    plant_hide_rate = traits.Float(0.0, desc='If the plant is visible, specifies a percentage of trials where it will be hidden')
    plant_type_options = list(plantlist.keys())
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))
    plant_visible = traits.Bool(True, desc='Specifies whether entire plant is displayed or just endpoint')
    cursor_radius = traits.Float(.5, desc="Radius of cursor")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor_visible = True

        # Initialize the plant
        if not hasattr(self, 'plant'):
            self.plant = plantlist[self.plant_type]
        self.plant_vis_prev = True

        # Add graphics models for the plant and targets to the window
        if hasattr(self.plant, 'graphics_models'):
            for model in self.plant.graphics_models:
                self.add_model(model)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self._target_color)
            target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self._target_color)

            self.targets = [target1, target2]
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)

        # Initialize target location variable
        self.target_location = np.array([0, 0, 0])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)

    def init(self):
        self.add_dtype('target', 'f8', (3,))
        self.add_dtype('target_index', 'i', (1,))
        super().init()

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen. Runs 60x per second.
        '''
        self.task_data['target'] = self.target_location.copy()
        self.task_data['target_index'] = self.target_index

        ## Run graphics commands to show/hide the plant if the visibility has changed
        if self.plant_type != 'CursorPlant':
            if self.plant_visible != self.plant_vis_prev:
                self.plant_vis_prev = self.plant_visible
                self.plant.set_visibility(self.plant_visible)
                # self.show_object(self.plant, show=self.plant_visible)

        self.move_effector()

        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

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
        try:
            super().run()
        finally:
            self.plant.stop()

    ##### HELPER AND UPDATE FUNCTIONS ####
    def update_cursor_visibility(self):
        ''' Update cursor visible flag to hide cursor if there has been no good data for more than 3 frames in a row'''
        prev = self.cursor_visible
        if self.no_data_count < 3:
            self.cursor_visible = True
            if prev != self.cursor_visible:
                self.show_object(self.cursor, show=True)
        else:
            self.cursor_visible = False
            if prev != self.cursor_visible:
                self.show_object(self.cursor, show=False)


    #### TEST FUNCTIONS ####
    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        return d <= (self.target_radius - self.cursor_radius)

    def _test_leave_early(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        rad = self.target_radius - self.cursor_radius
        return d > rad

    #### STATE FUNCTIONS ####
    def _start_wait(self):
        super()._start_wait()
        # hide targets
        for target in self.targets:
            target.hide()

    def _start_target(self):
        super()._start_target()

        # move one of the two targets to the new target location
        target = self.targets[self.target_index % 2]
        target.move_to_position(self.target_location)
        target.cue_trial_start()

    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)
        if idx < self.chain_length:
            target = self.targets[idx % 2]
            target.move_to_position(self.targs[idx])

    def _end_hold(self):
        # change current target color to green
        self.targets[self.target_index % 2].cue_trial_end_success()

    def _start_hold_penalty(self):
        super()._start_hold_penalty()
        # hide targets
        for target in self.targets:
            target.hide()

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        # hide targets
        for target in self.targets:
            target.hide()

    def _start_targ_transition(self):
        #hide targets
        for target in self.targets:
            target.hide()

    def _start_reward(self):
        self.targets[self.target_index % 2].show()

    #### Generator functions ####
    @staticmethod
    def centerout_2D_discrete(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10):
        '''

        Generates a sequence of 2D (x and z) target pairs with the first target
        always at the origin.

        Parameters
        ----------
        length : int
            The number of target pairs in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between the targets in a pair.

        Returns
        -------
        pairs : [nblocks*ntargets x 2 x 3] array of pairs of target locations


        '''

        # Choose a random sequence of points on the edge of a circle of radius
        # "distance"

        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)

        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T

        return pairs
