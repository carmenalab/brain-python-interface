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
from riglib import plants

from riglib.stereo_opengl.window import Window, FPScontrol, WindowDispl2D
from .target_graphics import *

## Plants
# List of possible "plants" that a subject could control either during manual or brain control
cursor_14x14 = plants.CursorPlant(endpt_bounds=(-14, 14, 0., 0., -14, 14))

shoulder_anchor = np.array([2., 0., -15])
chain_kwargs = dict(link_radii=.6, joint_radii=0.6, joint_colors=(181/256., 116/256., 96/256., 1), link_colors=(181/256., 116/256., 96/256., 1))

chain_20_20_endpt = plants.EndptControlled2LArm(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)
init_pos = np.array([0, 0, 0], np.float64)
chain_20_20_endpt.set_intrinsic_coordinates(init_pos)

chain_20_20 = plants.RobotArmGen2D(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)
init_pos = np.array([ 0.38118002,  2.08145271])
chain_20_20.set_intrinsic_coordinates(init_pos)

plantlist = dict(
    cursor_14x14=cursor_14x14,
    chain_20_20=chain_20_20,
    chain_20_20_endpt=chain_20_20_endpt)

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

        # index of current trial to keep track of target location
        self.trial_index = -1

        # index of current target presented to subject
        self.target_index = -1

        # number of targets to be acquired in this trial
        self.chain_length = len(self.targs)

        # Update the report stat on trials
        self.reportstats['Trial #'] = self.calc_trial_num()

    def _parse_next_trial(self):
        '''Check that the generator has the required data'''
        self.trial_index, self.targs = self.next_trial

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

class ScreenTargetCapture(TargetCapture, Window):
    """Concrete implementation of TargetCapture task where targets
    are acquired by "holding" a cursor in an on-screen target"""

    limit2d = 1

    sequence_generators = [
        'out_2D', 'centerout_2D', 'centeroutback_2D', 'rand_target_chain_2D', 'rand_target_chain_3D',
    ]

    hidden_traits = Window.hidden_traits + ['target_color', 'background', 'plant_hide_rate', 'starting_pos']

    is_bmi_seed = True

    # Runtime settable traits
    reward_time = traits.Float(.5, desc="Length of juice reward")
    target_radius = traits.Float(2, desc="Radius of targets in cm")
    target_color = traits.OptionsList(tuple(target_colors.keys()), desc="Color of the target")
    background = traits.Tuple((0,0,0,1), desc='Screen background color')

    hold_time = traits.Float(.2, desc="Length of hold required at targets")
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")
    timeout_penalty_time = traits.Float(1, desc="Length of penalty time for timeout error")
    max_attempts = traits.Int(10, desc='The number of attempts at a target before\
        skipping to the next one')

    plant_hide_rate = traits.Float(0.0, desc='If the plant is visible, specifies a percentage of trials where it will be hidden')
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))
    plant_visible = traits.Bool(True, desc='Specifies whether entire plant is displayed or just endpoint')
    cursor_radius = traits.Float(.5, desc='Radius of cursor in cm')
    starting_pos = traits.Tuple((5, 0, 5), desc='Where to initialize the cursor') 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the plant
        if not hasattr(self, 'plant'):
            self.plant = plantlist[self.plant_type]
        self.plant.set_endpoint_pos(np.array(self.starting_pos))
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
            target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
            target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])

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
        self.add_dtype('trial', 'i', (1,))
        self.add_dtype('trial_index', 'i', (1,))
        self.add_dtype('target', 'f8', (3,))
        self.add_dtype('target_index', 'i', (1,))
        self.add_dtype('plant_visible', '?', (1,))
        super().init()

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen
        '''
        self.task_data['trial'] = self.calc_trial_num()
        self.task_data['trial_index'] = self.trial_index
        self.task_data['target'] = self.target_location.copy()
        self.task_data['target_index'] = self.target_index

        self.move_effector()

        ## Run graphics commands to show/hide the plant if the visibility has changed
        self.update_plant_visibility()
        self.task_data['plant_visible'] = self.plant_visible

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
    def update_plant_visibility(self):
        ''' Update plant visibility'''
        if self.plant_visible != self.plant_vis_prev:
            self.plant_vis_prev = self.plant_visible
            self.plant.set_visibility(self.plant_visible)

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

        # stop the display sync if there is one
        self.sync_every_cycle = False    

        # hide targets
        for target in self.targets:
            target.hide()
            target.reset()

    def _start_target(self):
        super()._start_target()

        # start the display sync if there is one
        self.sync_every_cycle = True
        
        # move one of the two targets to the new target location
        target = self.targets[self.target_index % 2]
        target.move_to_position(self.target_location)
        target.show()

    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length:
            target = self.targets[next_idx % 2]
            target.move_to_position(self.targs[next_idx])
            target.show()

    def _start_targ_transition(self):
        super()._start_targ_transition()
        # hide the current target if there are more
        if self.target_index + 1 < self.chain_length:
            self.targets[self.target_index % 2].hide()

    def _start_hold_penalty(self):
        super()._start_hold_penalty()
        # hide targets
        for target in self.targets:
            target.hide()

        # stop the display sync
        self.sync_every_cycle = False

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        # hide targets
        for target in self.targets:
            target.hide()
        
        # stop the display sync
        self.sync_every_cycle = False

    def _start_reward(self):
        self.targets[self.target_index % 2].cue_trial_end_success()

    #### Generator functions ####
    @staticmethod
    def static(pos=(0,0,0), ntrials=0):
        '''Single location, finite (ntrials!=0) or infinite (ntrials==0)'''
        if ntrials == 0:
            while True:
                yield np.array(pos)
        else:
            for _ in range(ntrials):
                yield np.array(pos)

    @staticmethod
    def out_2D(nblocks=100, ntargets=8, distance=10, origin=(0,0,0)):
        '''
        Generates a sequence of 2D (x and z) targets at a given distance from the origin

        Parameters
        ----------
        nblocks : int
            The number of ntarget pairs in the sequence.
        ntargets : int
            The number of equally spaced targets
        distance : float
            The distance in cm between the center and peripheral targets.
        origin : 3-tuple
            Location of the central targets around which the peripheral targets span

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [1 x 3] target coordinates

        '''
        rng = np.random.default_rng()
        for _ in range(nblocks):
            order = np.arange(ntargets)
            rng.shuffle(order)
            for t in range(ntargets):
                idx = order[t]
                theta = 2*np.pi*idx/ntargets
                pos = np.array([
                    distance*np.cos(theta),
                    0,
                    distance*np.sin(theta)
                ]).T
                yield idx, [pos + origin]

    @staticmethod
    def centerout_2D(nblocks=100, ntargets=8, distance=10, origin=(0,0,0)):
        '''
        Pairs of central targets at the origin and peripheral targets centered around the origin

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [2 x 3] target coordinates
        '''
        gen = ScreenTargetCapture.out_2D(nblocks, ntargets, distance, origin)
        for _ in range(nblocks*ntargets):
            idx, pos = next(gen)
            targs = np.zeros([2, 3]) + origin
            targs[1,:] = pos[0]
            yield idx, targs

    @staticmethod
    def centeroutback_2D(nblocks=100, ntargets=8, distance=10, origin=(0,0,0)):
        '''
        Triplets of central targets, peripheral targets, and central targets

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [3 x 3] target coordinates
        '''
        gen = ScreenTargetCapture.out_2D(nblocks, ntargets, distance, origin)
        for _ in range(nblocks*ntargets):
            idx, pos = next(gen)
            targs = np.zeros([3, 3]) + origin
            targs[1,:] = pos[0]
            yield idx, targs
    
    @staticmethod
    def rand_target_chain_2D(ntrials=100, chain_length=1, boundaries=(-12,12,-12,12)):
        '''
        Generates a sequence of 2D (x and z) target pairs.

        Parameters
        ----------
        ntrials : int
            The number of target chains in the sequence.
        chain_length : int
            The number of targets in each chain
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)

        Returns
        -------
        [ntrials x chain_length x 3] array of target coordinates
        '''
        rng = np.random.default_rng()
        for t in range(ntrials):

            # Choose a random sequence of points within the boundaries
            pts = rng.uniform(size=(1, 3))*((boundaries[1]-boundaries[0]),
                0, (boundaries[3]-boundaries[2]))
            pts = pts+(boundaries[0], 0, boundaries[2])
            yield t, pts
    
    @staticmethod
    def rand_target_chain_3D(ntrials=100, chain_length=1, boundaries=(-12,12,-10,10,-12,12)):
        '''
        Generates a sequence of 3D target pairs.
        Parameters
        ----------
        ntrials : int
            The number of target chains in the sequence.
        chain_length : int
            The number of targets in each chain
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -y, y, -z, z)

        Returns
        -------
        [ntrials x chain_length x 3] array of target coordinates
        '''
        rng = np.random.default_rng()
        for t in range(ntrials):

            # Choose a random sequence of points within the boundaries
            pts = rng.uniform(size=(1, 3))*((boundaries[1]-boundaries[0]),
                (boundaries[3]-boundaries[2]), (boundaries[5]-boundaries[4]))
            pts = pts+(boundaries[0], boundaries[2], boundaries[4])
            return t, pts
