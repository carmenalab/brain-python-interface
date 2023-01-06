'''
A generic target capture task
'''
import numpy as np


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

class TargetCapture(Sequence):
    '''
    This is a generic cued target capture skeleton, to form as a common ancestor to the most
    common type of motor control task.
    '''
    status = dict(
        wait = dict(start_trial="target"),
        target = dict(enter_target="hold", timeout="timeout_penalty"),
        hold = dict(leave_target="hold_penalty", hold_complete="delay"),
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", end_state=True),
        hold_penalty = dict(hold_penalty_end="targ_transition", end_state=True),
        delay_penalty = dict(delay_penalty_end="targ_transition", end_state=True),
        reward = dict(reward_end="wait", stoppable=False, end_state=True)
    )

    # initial state
    state = "wait"

    target_index = -1 # Helper variable to keep track of which target to display within a trial
    tries = 0 # Helper variable to keep track of the number of failed attempts at a given trial.

    sequence_generators = []

    reward_time = traits.Float(.5, desc="Length of reward dispensation")
    hold_time = traits.Float(.2, desc="Length of hold required at targets before next target appears")
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    delay_time = traits.Float(0, desc="Length of time after a hold while the next target is on before the go cue")
    delay_penalty_time = traits.Float(1, desc="Length of penalty time for delay error")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")
    timeout_penalty_time = traits.Float(1, desc="Length of penalty time for timeout error")
    max_attempts = traits.Int(10, desc='The number of attempts of a target chain before\
        skipping to the next one')
    num_targets_per_attempt = traits.Int(2, desc="Minimum number of target acquisitions to be counted as an attempt")

    def init(self):
        self.trial_dtype = np.dtype([('trial', 'u4'), ('index', 'u4'), ('target', 'f8', (3,))])
        super().init()

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
        self.gen_indices, self.targs = self.next_trial
        # TODO error checking
        
        # Update the data sinks with trial information
        self.trial_record['trial'] = self.calc_trial_num()
        for i in range(len(self.gen_indices)):
            self.trial_record['index'] = self.gen_indices[i]
            self.trial_record['target'] = self.targs[i]
            self.sinks.send("trials", self.trial_record)

    def _start_target(self):
        self.target_index += 1

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

    def _start_delay(self):
        '''Nothing generic to do.'''
        pass

    def _while_delay(self):
        '''Nothing generic to do.'''
        pass

    def _end_delay(self):
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

    def _increment_tries(self):
        if self.target_index >= self.num_targets_per_attempt-1:
            self.tries += 1 # only count errors if the minimum number of targets have been acquired
        self.target_index = -1

        if self.tries < self.max_attempts: 
            self.trial_record['trial'] += 1
            for i in range(len(self.gen_indices)):
                self.trial_record['index'] = self.gen_indices[i]
                self.trial_record['target'] = self.targs[i]
                self.sinks.send("trials", self.trial_record)

    def _start_timeout_penalty(self):
        self._increment_tries()

    def _while_timeout_penalty(self):
        '''Nothing generic to do.'''
        pass

    def _end_timeout_penalty(self):
        '''Nothing generic to do.'''
        pass

    def _start_hold_penalty(self):
        self._increment_tries()

    def _while_hold_penalty(self):
        '''Nothing generic to do.'''
        pass

    def _end_hold_penalty(self):
        '''Nothing generic to do.'''
        pass

    def _start_delay_penalty(self):
        self._increment_tries()

    def _while_delay_penalty(self):
        '''Nothing generic to do.'''
        pass

    def _end_delay_penalty(self):
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
        return time_in_state > self.timeout_time or self.pause

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

    def _test_delay_complete(self, time_in_state):
        '''
        Test whether the delay period, when the cursor must stay in place
        while another target is being presented, is over. There should be 
        no delay on the last target in a chain.
        '''
        return self.target_index + 1 == self.chain_length or time_in_state > self.delay_time

    def _test_trial_complete(self, time_in_state):
        '''Test whether all targets in sequence have been acquired'''
        return self.target_index == self.chain_length-1

    def _test_trial_abort(self, time_in_state):
        '''Test whether the target capture sequence should just be skipped due to too many failures'''
        return (not self._test_trial_complete(time_in_state)) and (self.tries==self.max_attempts)

    def _test_trial_incomplete(self, time_in_state):
        '''Test whether the target capture sequence needs to be restarted'''
        return (not self._test_trial_complete(time_in_state)) and (self.tries<self.max_attempts) and not self.pause

    def _test_timeout_penalty_end(self, time_in_state):
        return time_in_state > self.timeout_penalty_time

    def _test_hold_penalty_end(self, time_in_state):
        return time_in_state > self.hold_penalty_time

    def _test_delay_penalty_end(self, time_in_state):
        return time_in_state > self.delay_penalty_time

    def _test_reward_end(self, time_in_state):
        return time_in_state > self.reward_time

    def _test_enter_target(self, time_in_state):
        '''This function is task-specific and not much can be done generically'''
        return False

    def _test_leave_target(self, time_in_state):
        '''This function is task-specific and not much can be done generically'''
        return self.pause

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

    limit2d = traits.Bool(True, desc="Limit cursor movement to 2D")

    sequence_generators = [
        'out_2D', 'centerout_2D', 'centeroutback_2D', 'rand_target_chain_2D', 'rand_target_chain_3D', 'corners_2D',
    ]

    hidden_traits = ['cursor_color', 'target_color', 'cursor_bounds', 'cursor_radius', 'plant_hide_rate', 'starting_pos']

    is_bmi_seed = True

    # Runtime settable traits
    target_radius = traits.Float(2, desc="Radius of targets in cm")
    target_color = traits.OptionsList("yellow", *target_colors, desc="Color of the target", bmi3d_input_options=list(target_colors.keys()))
    plant_hide_rate = traits.Float(0.0, desc='If the plant is visible, specifies a percentage of trials where it will be hidden')
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))
    plant_visible = traits.Bool(True, desc='Specifies whether entire plant is displayed or just endpoint')
    cursor_radius = traits.Float(.5, desc='Radius of cursor in cm')
    cursor_color = traits.OptionsList("dark_purple", *target_colors, desc='Color of cursor endpoint', bmi3d_input_options=list(target_colors.keys()))
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
            target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
            target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])

            self.targets = [target1, target2]

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

    #### TEST FUNCTIONS ####
    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[self.target_index])
        return d <= (self.target_radius - self.cursor_radius) or super()._test_enter_target(ts)

    def _test_leave_target(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[self.target_index])
        rad = self.target_radius - self.cursor_radius
        return d > rad or super()._test_leave_target(ts)

    #### STATE FUNCTIONS ####
    def _start_wait(self):
        super()._start_wait()

        if self.calc_trial_num() == 0:

            # Instantiate the targets here so they don't show up in any states that might come before "wait"
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

    def _start_target(self):
        super()._start_target()

        # Show target if it is hidden (this is the first target, or previous state was a penalty)
        target = self.targets[self.target_index % 2]
        if self.target_index == 0:
            target.move_to_position(self.targs[self.target_index])
            target.show()
            self.sync_event('TARGET_ON', self.gen_indices[self.target_index])

    def _start_hold(self):
        super()._start_hold()
        self.sync_event('CURSOR_ENTER_TARGET', self.gen_indices[self.target_index])

    def _start_delay(self):
        super()._start_delay()

        # Make next target visible unless this is the final target in the trial
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length:
            target = self.targets[next_idx % 2]
            target.move_to_position(self.targs[next_idx])
            target.show()
            self.sync_event('TARGET_ON', self.gen_indices[next_idx])
        else:
            # This delay state should only last 1 cycle, don't sync anything
            pass

    def _start_targ_transition(self):
        super()._start_targ_transition()
        if self.target_index == -1:

            # Came from a penalty state
            pass
        elif self.target_index + 1 < self.chain_length:

            # Hide the current target if there are more
            self.targets[self.target_index % 2].hide()
            self.sync_event('TARGET_OFF', self.gen_indices[self.target_index])

    def _start_hold_penalty(self):
        self.sync_event('HOLD_PENALTY') 
        super()._start_hold_penalty()
        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

    def _end_hold_penalty(self):
        super()._end_hold_penalty()
        self.sync_event('TRIAL_END')

    def _start_delay_penalty(self):
        self.sync_event('DELAY_PENALTY') 
        super()._start_delay_penalty()
        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

    def _end_delay_penalty(self):
        super()._end_delay_penalty()
        self.sync_event('TRIAL_END')
        
    def _start_timeout_penalty(self):
        self.sync_event('TIMEOUT_PENALTY')
        super()._start_timeout_penalty()
        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

    def _end_timeout_penalty(self):
        super()._end_timeout_penalty()
        self.sync_event('TRIAL_END')

    def _start_reward(self):
        self.targets[self.target_index % 2].cue_trial_end_success()
        self.sync_event('REWARD')
        
    
    def _end_reward(self):
        super()._end_reward()
        self.sync_event('TRIAL_END')

        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

    #### Generator functions ####
    '''
    Note to self: because of the way these get into the database, the parameters don't
    have human-readable descriptions like the other traits. So it is useful to define
    the descriptions elsewhere, in models.py under Generator.to_json().

    Ideally someone should take the time to reimplement generators as their own classes
    rather than static methods that belong to a task.
    '''
    @staticmethod
    def static(pos=(0,0,0), ntrials=0):
        '''Single location, finite (ntrials!=0) or infinite (ntrials==0)'''
        if ntrials == 0:
            while True:
                yield [0], np.array(pos)
        else:
            for _ in range(ntrials):
                yield [0], np.array(pos)

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
            order = np.arange(ntargets) + 1 # target indices, starting from 1
            rng.shuffle(order)
            for t in range(ntargets):
                idx = order[t]
                theta = 2*np.pi*(3-idx)/ntargets # put idx 1 at 12 o'clock
                pos = np.array([
                    distance*np.cos(theta),
                    0,
                    distance*np.sin(theta)
                ]).T
                yield [idx], [pos + origin]

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
            indices = np.zeros([2,1])
            indices[1] = idx
            yield indices, targs

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
            indices = np.zeros([3,1])
            indices[1] = idx
            yield indices, targs
    
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
        boundaries: 4 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)

        Returns
        -------
        [ntrials x chain_length x 3] array of target coordinates
        '''
        rng = np.random.default_rng()
        idx = 0
        for t in range(ntrials):

            # Choose a random sequence of points within the boundaries
            pts = rng.uniform(size=(chain_length, 3))*((boundaries[1]-boundaries[0]),
                0, (boundaries[3]-boundaries[2]))
            pts = pts+(boundaries[0], 0, boundaries[2])
            yield idx+np.arange(chain_length), pts
            idx += chain_length
    
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
        idx = 0
        for t in range(ntrials):

            # Choose a random sequence of points within the boundaries
            pts = rng.uniform(size=(chain_length, 3))*((boundaries[1]-boundaries[0]),
                (boundaries[3]-boundaries[2]), (boundaries[5]-boundaries[4]))
            pts = pts+(boundaries[0], boundaries[2], boundaries[4])
            yield idx+np.arange(chain_length), pts
            idx += chain_length

    @staticmethod
    def corners_2D(nblocks=5, chain_length=1, corners=(-8,8,-8,8)):
        '''
        Generates a sequence of 2D (x and z) targets at the given 4 corners

        Parameters
        ----------
        nblocks : 3-tuple
            Number of blocks
        chain_length : int
            The number of targets in each chain before a reward is given
        corners : 4-tuple
            Location of the edges of the screen (-x, x, -y, y)

        Returns
        -------
        [nblocks*4 x 1] array of tuples containing trial indices and [1 x 3] target coordinates

        '''
        ntargets = 4
        corners = np.array([
            [corners[0], 0, corners[2]],
            [corners[0], 0, corners[3]],
            [corners[1], 0, corners[2]],
            [corners[1], 0, corners[3]]
        ])
        target_order = []
        rng = np.random.default_rng()
        for _ in range(nblocks):
            order = np.arange(ntargets) + 1 # target indices, starting from 1
            rng.shuffle(order)
            target_order = np.concatenate((target_order, order), axis=0)

        # Spit out trials in groups of chain_length
        ntrials = nblocks*4//chain_length
        for t in range(ntrials):
            idx = target_order[int(t*chain_length):int(t*chain_length+chain_length)]
            pos = [corners[int(i-1),:] for i in idx]
            yield idx, pos

class ScreenReachAngle(ScreenTargetCapture):
    '''
    A modified task that requires the cursor to move in the right direction towards the target, 
    without actually needing to arrive at the target. If the maximum angle is exceeded, a reach 
    penalty is applied. No hold or delay period.

    Only works for sequences with 1 target in a chain. 
    '''

    status = dict(
        wait = dict(start_trial="target"),
        target = dict(reach_success="targ_transition", timeout="timeout_penalty", leave_bounds="reach_penalty"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", end_state=True),
        reach_penalty = dict(reach_penalty_end="targ_transition", end_state=True),
        reward = dict(reward_end="wait", stoppable=False, end_state=True)
    )

    sequence_generators = [
        'out_2D', 'rand_target_chain_2D', 'rand_target_chain_3D', 'discrete_targets_2D',
    ]

    max_reach_angle = traits.Float(90., desc="Angle defining the boundaries between the starting position of the cursor and the target")
    reach_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    reach_fraction = traits.Float(0.5, desc="Fraction of the distance between the reach start and the target before a reward")
    start_radius = traits.Float(1., desc="Buffer around reach start allowed in bounds (cm)")

    exclude_parent_traits = ['hold_time', 'hold_penalty_time', 'delay_time', 'delay_penalty_time']

    def _start_target(self):
        super()._start_target()

        # Define a reach start and reach target position whenever the target appears
        self.reach_start = self.plant.get_endpoint_pos().copy()
        self.reach_target = self.targs[self.target_index]

    def _test_leave_bounds(self, ts):
        '''
        Check whether the cursor is in the boundary defined by reach_start, target_pos,
        and max_reach_angle.
        '''

        # Calculate the angle between the vectors from the start pos to the current cursor and target
        a = self.plant.get_endpoint_pos() - self.reach_start
        b = self.reach_target - self.reach_start
        cursor_target_angle = np.arccos(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b))

        # If that angle is more than half the maximum, we are outside the bounds
        out_of_bounds = np.degrees(cursor_target_angle) > self.max_reach_angle / 2

        # But also allow a target radius around the reach_start 
        away_from_start = np.linalg.norm(self.plant.get_endpoint_pos() - self.reach_start) > self.start_radius

        return away_from_start and out_of_bounds

    def _test_reach_success(self, ts):
        dist_traveled = np.linalg.norm(self.plant.get_endpoint_pos() - self.reach_start)
        dist_total = np.linalg.norm(self.reach_target - self.reach_start)
        dist_total -= (self.target_radius - self.cursor_radius)
        return dist_traveled/dist_total > self.reach_fraction

    def _start_reach_penalty(self):
        self.sync_event('OTHER_PENALTY')
        self._increment_tries()
        
        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

    def _end_reach_penalty(self):
        self.sync_event('TRIAL_END')

    def _test_reach_penalty_end(self, ts):
        return ts > self.reach_penalty_time

    @staticmethod
    def discrete_targets_2D(nblocks=100, ntargets=3, boundaries=(-6,6,-3,3)):
        '''
        Generates a sequence of 2D (x and z) target pairs that don't overlap

        Parameters
        ----------
        nblocks : int
            The number of ntarget pairs in the sequence.
        ntargets : int
            The number of unique targets (up to 9 maximum)
        boundaries: 4 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)

        Returns
        -------
        [ntrials x ntargets x 3] array of target coordinates
        '''
        targets = np.array([
            [0, 0.5],
            [1, 0.5],
            [1, 0],
            [0, 0],
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
            [0.5, 1],
        ])
        rng = np.random.default_rng()
        for _ in range(nblocks):
            order = np.arange(ntargets) # target indices
            rng.shuffle(order)
            for t in range(ntargets):
                idx = order[t]
                pts = targets[idx]*((boundaries[1]-boundaries[0]),
                    (boundaries[3]-boundaries[2]))
                pts = pts+(boundaries[0], boundaries[2])
                pos = np.array([pts[0], 0, pts[1]])
                yield [idx], [pos]