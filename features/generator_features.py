'''
Features which have task-like functionality w.r.t. task...
'''

import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from riglib.experiment import traits
from built_in_tasks.target_graphics import *

class Autostart(traits.HasTraits):
    '''
    Automatically begins the trial from the wait state, 
    with a random interval drawn from `rand_start`. Doesn't really
    work if there are multiple trials in between wait states.
    '''
    rand_start = traits.Tuple((0., 0.), desc="Start interval")
    exclude_parent_traits = ['wait_time']

    def _start_wait(self):
        '''
        At the start of the 'wait' state, determine how long to wait before starting the trial
        by drawing a sample from the rand_start interval
        '''
        s, e = self.rand_start
        self.wait_time = random.random()*(e-s) + s
        super(Autostart, self)._start_wait()
        
    def _test_start_trial(self, ts):
        '''
        Test if the required random wait time has passed
        '''
        return ts > self.wait_time and not self.pause

class AdaptiveGenerator(object):
    '''
    Deprecated--this class appears to be unused
    '''
    def __init__(self, *args, **kwargs):
        super(AdaptiveGenerator, self).__init__(*args, **kwargs)
        assert hasattr(self.gen, "correct"), "Must use adaptive generator!"

    def _start_reward(self):
        self.gen.correct()
        super(AdaptiveGenerator, self)._start_reward()
    
    def _start_incorrect(self):
        self.gen.incorrect()
        super(AdaptiveGenerator, self)._start_incorrect()


class IgnoreCorrectness(object):
    '''Deprecated--this class appears to be unused and not compatible with Sequences
    Allows any response to be correct, not just the one defined. Overrides for trialtypes'''
    def __init__(self, *args, **kwargs):
        super(IgnoreCorrectness, self).__init__(*args, **kwargs)
        if hasattr(self, "trial_types"):
            for ttype in self.trial_types:
                del self.status[ttype]["%s_correct"%ttype]
                del self.status[ttype]["%s_incorrect"%ttype]
                self.status[ttype]["correct"] = "reward"
                self.status[ttype]["incorrect"] = "penalty"

    def _test_correct(self, ts):
        return self.event is not None

    def _test_incorrect(self, ts):
        return False


class MultiHoldTime(traits.HasTraits):
    '''
    Deprecated--Use RandomDelay instead. 
    Allows the hold time parameter to be multiple values per target in a given sequence chain. For instance,
    center targets and peripheral targets can have different hold times.
    '''

    multi_hold_time = traits.List([.2,], desc="Length of hold required at targets before next target appears. \
        Can be a single number or a list of numbers to apply to each target in the sequence (center, out, etc.)")
    exclude_parent_traits = ['hold_time']
    
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
        if len(self.multi_hold_time) == 1:
            multi_hold_time = self.multi_hold_time[0]
        else:
            multi_hold_time = self.multi_hold_time[self.target_index]
        return time_in_state > multi_hold_time

class Progressbar_fixation(traits.HasTraits):

    progress_bar_color = traits.OptionsList("blue", *target_colors, desc="Color of the eye target", bmi3d_input_options=list(target_colors.keys()))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:

            # Target 1 and 2 are for saccade. Target 3 is for hand
            self.bar = VirtualCircularTarget(target_radius=0.5, target_color=target_colors[self.progress_bar_color], starting_pos=[-5,0,-5])

    def _start_wait(self):
        super()._start_wait()

        if self.calc_trial_num() == 0:

            for model in self.bar.graphics_models:
                self.add_model(model)
                self.bar.hide()

    def _start_fixation(self):
        super()._start_fixation()
        self.fixation_frame_index = 0

    def _while_fixation(self):
        super()._while_fixation()
        self.fixation_frame_index += 1
        fixation_time = self.fixation_frame_index*2/self.fps
        if fixation_time <= self.hold_time:
            radius = fixation_time/self.hold_time*(self.fixation_radius-self.fixation_radius_buffer)
        else:
            radius = (self.fixation_radius-self.fixation_radius_buffer)

        if hasattr(self, 'bar'):
            for model in self.bar.graphics_models:
                self.remove_model(model)
            del self.bar

        self.bar = VirtualCircularTarget(target_radius=radius, target_color=target_colors[self.progress_bar_color], \
                                         starting_pos=self.targs[self.target_index]+[0,-0.5,0]) # [0,-0.5,0] is for visualization
        for model in self.bar.graphics_models:
            self.add_model(model)
        self.bar.show()

    def _start_targ_transition(self):
        super()._start_targ_transition()
        self.bar.hide()


class RandomDelay(traits.HasTraits):
    '''
    Replaces 'delay_time' with 'rand_delay', an interval on which the delay period is selected uniformly.
    '''
    
    rand_delay = traits.Tuple((0., 0.), desc="Delay interval")
    exclude_parent_traits = ['delay_time']
    prob_catch_trials = traits.Float(0., desc="Probability of catch trials")
    short_delay_catch_trials = traits.List([.2,], desc="Delay intervals for catch trials")

    def _start_wait(self):
        '''
        At the start of the 'wait' state, draw a sample from the rand_delay interval for this trial.
        '''

        # only make new delay_time when the previous state was not pause or penalty
        if self.pause_index == 0 and self.penalty_index == 0:
            # Catch trial condition
            if random.random() < self.prob_catch_trials:
                self.delay_time = random.choice(self.short_delay_catch_trials)

            # Normal trial condition
            else:
                s, e = self.rand_delay
                self.delay_time = random.random()*(e-s) + s
        
        super()._start_wait()

class DiscreteRandomDelay_EyeHandSequence(traits.HasTraits):
    '''
    Choose delay time from three discrete blocks (short, meidum, long delay) in eye-hand sequence task
    '''
    
    rand_delay_hand_lower_bound = traits.List([0,], desc="lower bound of delay time in each block for hand go cue in sequence trials")
    rand_delay_hand_upper_bound = traits.List([0.1,], desc="upper bound of delay time in each block for hand go cue in sequence trials")
    rand_delay_hand_probability = traits.List([1.,], desc="probablity of each block for delay time")
    exclude_parent_traits = ['rand_delay_hand']

    def _start_wait(self):
        super()._start_wait()

        if self.tries == 0:
            if self.trial_count_blocks - self.trials_block_simultaneous < self.trials_block_sequence: # only in sequence trials
                
                # Set delay time for hand go cue in sequence trials
                delays = []
                for lower, upper in zip(self.rand_delay_hand_lower_bound, self.rand_delay_hand_upper_bound):
                    delays.append(random.random()*(upper-lower) + lower)

                self.delay_time_hand = np.random.choice(delays, p = (self.rand_delay_hand_probability))

class TransparentDelayTarget(traits.HasTraits):
    '''
    Feature to make the delay period show a semi-transparent target rather than the full target. Used 
    for training the go cue. Gradually increase the alpha from 0 to 0.75 once a long enough delay 
    period has been established.
    '''

    delay_target_alpha = traits.Float(0.25, desc="Transparency of the next target during delay periods")

    def _start_delay(self):
        super()._start_delay()

        # Set the alpha of the next target
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length:
            target = self.targets[next_idx % 2]

            if hasattr(target, "sphere"):
                self._old_target_color = np.copy(target.sphere.color)
                new_target_color = list(target.sphere.color)
                new_target_color[3] = self.delay_target_alpha
                target.sphere.color = tuple(new_target_color)
            elif hasattr(target, "cube"):
                self._old_target_color = np.copy(target.cube.color)
                new_target_color = list(target.cube.color)
                new_target_color[3] = self.delay_target_alpha
                target.cube.color = tuple(new_target_color)

    def _start_target(self):
        super()._start_target()

        # Reset the transparency of the current target
        if self.target_index > 0:
            target = self.targets[self.target_index % 2]
            if hasattr(target, "sphere"):
                target.sphere.color = self._old_target_color
            elif hasattr(target, "cube"):
                target.cube.color = self._old_target_color

class StartTrialBelowSpeedThr(traits.HasTraits):
    '''
    The next trial doesn't start until the cursor speed is kept under the speed threshold for a certain duration
    Please turn off the Autostart feature. Otherwise, it overwrites this feature and the next trial automatically starts 
    '''

    exclude_parent_traits = ['wait_time','rand_start']
    speed_threshold = traits.Float(30, desc="Speed threshold in cm/s")
    duration_under_speed_threshold = traits.Float(1, desc="The duration in second in which the speed must stay below the speed threshold")

    def _start_wait(self):
        
        self.previous_cursor_pos = self.plant.get_endpoint_pos()
        self.frames_under_speed_threshold = 0
    
        super()._start_wait()

    def _test_start_trial(self, time_in_state):

        self.current_cursor_pos = self.plant.get_endpoint_pos()

        speed = np.linalg.norm(self.current_cursor_pos - self.previous_cursor_pos)*self.fps

        if speed < self.speed_threshold:
            self.frames_under_speed_threshold += 1
        else:
            self.frames_under_speed_threshold = 0

        self.previous_cursor_pos = self.current_cursor_pos

        return self.frames_under_speed_threshold/self.fps > self.duration_under_speed_threshold

class PoissonWait(traits.HasTraits):
    '''
    Draw each trial's wait time from a poisson random distribution    
    '''
    
    poisson_mu = traits.Float(0.5, desc="Mean duration between trials (s)")
    exclude_parent_traits = ['wait_time']

    def _parse_next_trial(self):
        self.wait_time = np.random.exponential(self.poisson_mu)
        super()._parse_next_trial()

class IncrementalRotation(traits.HasTraits):
    '''
    Gradually change the perturbation rotation over trials
    '''
    exclude_parent_traits = ['pertubation_rotation', 'perturbation_rotation_z', 'perturbation_rotation_x']

    init_rotation_y  = traits.Float(0.0, desc="initial rotation about bmi3d y-axis in degrees")
    final_rotation_y = traits.Float(0.0, desc="final rotation about bmi3d y-axis in degrees")

    init_rotation_z  = traits.Float(0.0, desc="initial rotation about bmi3d z-axis in degrees")
    final_rotation_z = traits.Float(0.0, desc="final rotation about bmi3d z-axis in degrees")
    
    init_rotation_x  = traits.Float(0.0, desc="inital rotation about bmi3d x-axis in degrees")
    final_rotation_x = traits.Float(0.0, desc="final rotation about bmi3d x-axis in degrees")

    delta_rotation_y = traits.Float(0.0, desc="rotation step size about bmi3d y-axis in degrees")
    delta_rotation_z = traits.Float(0.0, desc="rotation step size about bmi3d z-axis in degrees")
    delta_rotation_x = traits.Float(0.0, desc="rotation step size about bmi3d x-axis in degrees")

    trials_per_increment = traits.Int(1, desc="number of successful trials per rotation step")
    final_tracking_out_time = traits.Float(1.6, desc="Time allowed to be tracking outside the target for final rotation") # AKA tolerance time

    def init(self):    
        super().init()
        self.num_trials_success = 0
        self.num_increments_y = 0
        self.num_increments_z = 0
        self.num_increments_x = 0

        if self.final_rotation_y != self.init_rotation_y:
            self.num_increments_y = int( (self.final_rotation_y-self.init_rotation_y) / self.delta_rotation_y+1 )
        if self.final_rotation_z != self.init_rotation_z:
            self.num_increments_z = int( (self.final_rotation_z-self.init_rotation_z) / self.delta_rotation_z+1 )
        if self.final_rotation_x != self.init_rotation_x:
            self.num_increments_x = int( (self.final_rotation_x-self.init_rotation_x) / self.delta_rotation_x+1 )

        self.max_num_increments = np.max([self.num_increments_y, self.num_increments_z, self.num_increments_x])
        self.pertubation_rotation = self.init_rotation_y
        self.perturbation_rotation_z = self.init_rotation_z
        self.perturbation_rotation_x = self.init_rotation_x

        print("Y", self.pertubation_rotation, "Z", self.perturbation_rotation_z, "X", self.perturbation_rotation_x)
    
    def incremental_start_wait(self):
        # determine the current rotation step
        num_deltas = int(self.num_trials_success / self.trials_per_increment)

        # increment the current perturbation rotation by delta
        self.pertubation_rotation = self.init_rotation_y + self.delta_rotation_y*num_deltas
        self.perturbation_rotation_z = self.init_rotation_z + self.delta_rotation_z*num_deltas
        self.perturbation_rotation_x = self.init_rotation_x + self.delta_rotation_x*num_deltas

        # change tracking out time of final rotation
        if num_deltas+1 == self.max_num_increments:
            self.tracking_out_time = self.final_tracking_out_time

        # stop incrementing once final perturbation rotation reached
        if self.num_trials_success >= self.num_increments_y * self.trials_per_increment:
            self.pertubation_rotation = self.final_rotation_y
        if self.num_trials_success >= self.num_increments_z * self.trials_per_increment:
            self.perturbation_rotation_z = self.final_rotation_z
        if self.num_trials_success >= self.num_increments_x * self.trials_per_increment:
            self.perturbation_rotation_x = self.final_rotation_x
        
        print("Y", self.pertubation_rotation, "Z", self.perturbation_rotation_z, "X", self.perturbation_rotation_x)
        print(self.tracking_out_time, "tracking out")
    
    def _start_wait(self):
        super()._start_wait()
        self.incremental_start_wait()

    def _start_wait_retry(self):
        super()._start_wait_retry()
        self.incremental_start_wait()

    def _start_reward(self):
        super()._start_reward()
        self.num_trials_success += 1


class HideLeftTrajectory(traits.HasTraits):
    '''
    Hide the left side of the 1d target trajectory. 
    This will make only the 'lookahead' of the trajectory visible. The 'lookbehind' will be blacked out.
    Useful for task with bumpers.
    '''

    def setup_start_wait(self):
        super().setup_start_wait()
        self.trajectory.update_mask(self.lookahead, self.lookahead*2+1) # frame indices to show 
        # a total of lookahead*2+1 frames spans the screen width: lookahead index is at center with n_lookahead frames on either side
                                    
    def update_frame(self):
        self.trajectory.update_mask(self.lookahead+self.frame_index, self.lookahead*2+1+self.frame_index) # frame indices to show
        super().update_frame() # frame_index is incremented at the end of update_frame(), so need to update trajectory mask before this

class ReadysetMedley(traits.HasTraits):

    '''
    Allows for mulitple different prepbuff and delay times to be used within a single experiment.
    Replaces the prepbuff_time and delay_time parameters with a list of possible values and corresponding probabilities.
    '''

    exclude_parent_traits = ['delay_time']
    display_times = traits.List([0,], desc = 'Possible peripheral target display times')
    frac_times = traits.List([0.1,], desc = 'Proportion of each type of display time. Need to be equal length to delay_times and sum to 1') #should sum to 1.0 
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(len(self.display_times) == len(self.frac_times)) #can be here and task wont run if lenghts are different

    def _start_wait(self):
        '''
        At the start of the 'wait' state, determine which prepbuff & delay_time to use
        '''
        self.delay_time = np.random.choice((self.display_times), p = (self.frac_times)) 
        audio_length = 2*self.tone_space - self.early_move_time #use the same value but update to account for the different delay times
        self.prepbuff_time = audio_length - self.delay_time #update the prepbuff time accordingly
        super()._start_wait()

class ReadysetColorChange(traits.HasTraits):
    '''
    Change the color to go along with the set cue
    '''
    def color_set_cue(self): #update function to have color change 
        self.targets[0].cue_set_tone() #turn target orange on set tone

    def color_go_cue(self): #update function to have color change
        self.targets[0].show() #need to show as the target will be hidden by the baseline logic
        self.targets[0].cue_fixation() #turn target blue on go tone
                    
    def _start_tooslow_penalty(self):
        self.tooslow_start = self.get_time()
        super()._start_tooslow_penalty()
        self.targets[0].show()
        self.targets[0].cue_trial_end_failure()
        self.targets[1].hide()
    
    def _while_tooslow_penalty(self):
        if (self.get_time() - self.tooslow_start) >= 0.5 * self.tooslow_penalty_time:
            for targets in self.targets:
                targets.hide()
                targets.reset()

    def _end_tooslow_penalty(self):
        self.sync_event('TRIAL_END')

class HideCursorReturn(traits.HasTraits):

    '''
    Hide the cursor during the return to center after a reward or penalty. 
    Display again when cursor is within some user specified distance of center of the center target.

    '''
    show_cursor_return = traits.Float(3, desc = 'Distance from center at which to turn curson on')

    def _start_reward(self):
        super()._start_reward()
        self.plant_visible = False 
    
    def _while_target(self):

        if self.calc_trial_num() > 0: #skip logic for very first trial of block 
            if self.target_index == 0: 
                cursor_pos = self.plant.get_endpoint_pos()
                dist_from_center = np.linalg.norm(cursor_pos - self.targs[self.target_index])
                if dist_from_center < self.show_cursor_return:
                    self.plant_visible = True
                else:
                    self.plant_visible = False
    
    def _start_hold_penalty(self):
        super()._start_hold_penalty()
        self.plant_visible = False

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        self.plant_visible = False
    
    def _start_delay_penalty(self):
        super()._start_delay_penalty()
        self.plant_visible = False  

    def _start_tooslow_penalty(self):
        super()._start_tooslow_penalty()
        self.plant_visible = False
