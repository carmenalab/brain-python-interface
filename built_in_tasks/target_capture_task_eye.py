'''
Target capture tasks with eye position requirement
'''
import numpy as np
import random
import os

from .target_graphics import *
from riglib.stereo_opengl.window import Window
from .target_capture_task import ScreenTargetCapture
from riglib.experiment import traits, Sequence
from riglib import plants
from collections import deque
import aopy
import tables
import collections
import re

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

class EyeConstrainedTargetCapture(ScreenTargetCapture):
    '''
    Fixation requirement is added before go cue
    '''

    fixation_penalty_time = traits.Float(0., desc="Time in fixation penalty state")
    fixation_target_color = traits.OptionsList("fixation_color", *target_colors, desc="Color of the center target under fixation state", bmi3d_input_options=list(target_colors.keys()))
    eye_target_color = traits.OptionsList("eye_color", *target_colors, desc="Color of the center target under fixation state", bmi3d_input_options=list(target_colors.keys()))
    fixation_radius_buffer = traits.Float(.5, desc="additional radius for eye target")
    hand_target_color = traits.OptionsList("yellow", *target_colors, desc="Color for the hand-only target", bmi3d_input_options=list(target_colors.keys()))
    fixation_radius = traits.Float(.5, desc="additional radius for eye target")

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(timeout="timeout_penalty", gaze_enter_target="hold", start_pause="pause"),
        hold = dict(leave_target="hold_penalty", hold_complete="delay", fixation_break="fixation_penalty", start_pause="pause"),
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition", fixation_break="fixation_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:

            target3 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            self.targets_eye = [target3]
            self.offset_cube = np.array([0,10,self.fixation_radius/2]) # To center the cube target

    def _test_gaze_enter_target(self,ts):
        '''
        Check whether eye positions and hand cursor are within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])

        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.targs[self.target_index])
        
        # Fixation requirement is only for the center target
        if self.target_index == 0:
            return (eye_d <= self.target_radius + self.fixation_radius_buffer) and (hand_d <= self.target_radius - self.cursor_radius)
        else:
            return hand_d <= self.target_radius - self.cursor_radius
        
    def _test_fixation_break(self,ts):
        '''
        Triggers the fixation_penalty state when eye positions are outside fixation distance
        Only apply this to the first hold and delay period
        '''
        if self.target_index <= 0:   
            eye_d = np.linalg.norm(self.calibrated_eye_pos)
            return (eye_d > self.target_radius + self.fixation_radius_buffer)
    
    def _test_fixation_penalty_end(self,ts):
        return (ts > self.fixation_penalty_time) 

    def _start_wait(self):
        super()._start_wait()
        if self.calc_trial_num() == 0:

            # Instantiate the targets here so they don't show up in any states that might come before "wait"
            for target in self.targets_eye:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

    def _start_target(self):
        super()._start_target()

        if self.target_index == 0:
            self.targets_eye[0].move_to_position(self.targs[self.target_index] - self.offset_cube)
            self.targets_eye[0].show()

    def _while_target(self):
        if self.target_index == 0:
            eye_pos = self.calibrated_eye_pos
            eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])
            if eye_d <= (self.target_radius + self.fixation_radius_buffer):
                self.targets_eye[0].cube.color = target_colors[self.fixation_target_color] # chnage color in fixating center
            else:
                self.targets_eye[0].reset()

    def _start_delay(self):
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length: # This is for hand target in the second delay
            self.targets[next_idx].move_to_position(self.targs[next_idx])
            self.targets[next_idx].sphere.color = target_colors[self.hand_target_color]
            self.targets[next_idx].show()
            self.sync_event('TARGET_ON', self.gen_indices[next_idx])

    def _start_targ_transition(self):
        super()._start_targ_transition()

        if self.target_index + 1 < self.chain_length:
            # Hide the current target if there are more
            self.targets_eye[0].hide()

    def _start_hold_penalty(self):
        super()._start_hold_penalty()
        self.targets_eye[0].hide()
        self.targets_eye[0].reset()

    def _start_delay_penalty(self):
        super()._start_delay_penalty()
        self.targets_eye[0].hide()
        self.targets_eye[0].reset()

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        self.targets_eye[0].hide()
        self.targets_eye[0].reset()

    def _start_fixation_penalty(self):
        if hasattr(super(), '_start_fixation_penalty'):
            super()._start_fixation_penalty()

        self._increment_tries()
        self.sync_event('FIXATION_PENALTY') 
        self.penalty_index = 1
        self.num_fixation_state = 0

        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

        self.targets_eye[0].hide()
        self.targets_eye[0].reset()

    def _end_fixation_penalty(self):
        self.sync_event('TRIAL_END')

class HandConstrainedEyeCapture(ScreenTargetCapture):
    '''
    Saccade task with holding another target with hand. Subjects need to hold an initial target with their hand. 
    Then they need to fixate the first eye target and make a saccade for the second eye target. The eye target is a square whose width is adjusted by fixation_radius.
    The acceprance radius for eye fixation is fixation_radius (width) + fixation_radius_buffer. The buffer radius is invisible for subjects. 
    2 of chain_length is only tested.
    '''

    fixation_radius = traits.Float(2.5, desc="Width of the square eye target")
    fixation_penalty_time = traits.Float(1.0, desc="Time in fixation penalty state")
    fixation_target_color = traits.OptionsList("fixation_color", *target_colors, desc="Color of the eye target under fixation state", bmi3d_input_options=list(target_colors.keys()))
    eye_target_color = traits.OptionsList("eye_color", *target_colors, desc="Color of the eye target", bmi3d_input_options=list(target_colors.keys()))
    fixation_radius_buffer = traits.Float(.5, desc="additional radius for eye target. fixation_radius (width) + buffer determines the break of fixation")
    fixation_time = traits.Float(.2, desc="fixation duration during which subjects have to keep fixating the eye target")
    incorrect_target_radius_buffer = traits.Float(.5, desc="target radius + buffer radius determines if subjects look at the incorrect target")
    incorrect_target_penalty_time = traits.Float(1, desc="Length of penalty time for acquiring an incorrect target")

    exclude_parent_traits = ['hold_time']
    hidden_traits = ['eye_target_color', 'fixation_target_color']

    status = dict(
        wait = dict(start_trial="init_target", start_pause="pause"),
        init_target = dict(enter_target="target", start_pause="pause"),
        target = dict(start_pause="pause", timeout="timeout_penalty", return_init_target='init_target', gaze_enter_target="fixation"),
        target_eye = dict(start_pause="pause", timeout="timeout_penalty", leave_target='hold_penalty', gaze_target="fixation", gaze_incorrect_target="incorrect_target_penalty"),
        fixation = dict(start_pause="pause", leave_target="hold_penalty", fixation_complete="delay", fixation_break="fixation_penalty"), 
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition", fixation_break="fixation_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target_eye", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="wait", start_pause="pause", end_state=True),
        incorrect_target_penalty = dict(incorrect_target_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )
 
    sequence_generators = ['row_target','sac_hand_2d']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:

            # Target 1 and 2 are for saccade. Target 3 is for hand
            target1 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            target2 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            target3 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])

            self.targets = [target1, target2]
            self.targets_hand = [target3]

            self.offset_cube = np.array([0,0,self.fixation_radius/2]) # To center the cube target
        
    def _parse_next_trial(self):
        '''Check that the generator has the required data'''
        # 2 target positions for hand and eye. The first and second target index is for eye, and the third one is for hand
        self.gen_indices, self.targs = self.next_trial 

        # Update the data sinks with trial information
        self.trial_record['trial'] = self.calc_trial_num()
        for i in range(len(self.gen_indices)):
            self.trial_record['index'] = self.gen_indices[i]
            self.trial_record['target'] = self.targs[i]
            self.sinks.send("trials", self.trial_record)

    def _test_gaze_enter_target(self,ts):
        '''
        Check whether eye positions and hand cursor are within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])

        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.targs[-1])
        
        return (eye_d <= self.fixation_radius + self.fixation_radius_buffer) and (hand_d <= self.target_radius - self.cursor_radius)
    
    def _test_gaze_target(self, ts):
        '''
        Check whether eye position is within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])

        return eye_d <= self.fixation_radius + self.fixation_radius_buffer   
    
    def _test_gaze_incorrect_target(self, ts):
        '''
        Check whether eye position is within the different target (hand target)
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[-1,[0,2]])

        return eye_d <= self.target_radius + self.incorrect_target_radius_buffer
    
    def _test_fixation_break(self,ts):
        '''
        Triggers the fixation_penalty state when eye positions are outside fixation distance
        '''
        # Distance of an eye position from a target position
        eye_pos = self.calibrated_eye_pos
        d_eye = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])
        return d_eye > self.fixation_radius + self.fixation_radius_buffer
    
    def _test_fixation_complete(self,ts):
        return ts > self.fixation_time
    
    def _test_fixation_penalty_end(self,ts):
        return ts > self.fixation_penalty_time
    
    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[-1]) # hand must be within the initial target
        return d <= self.target_radius - self.cursor_radius

    def _test_leave_target(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[-1]) # hand must be within the initial target
        return d > self.target_radius - self.cursor_radius

    def _test_return_init_target(self, ts):
        '''
        return true if cursor moves outside the exit radius, but only applied when the target index is 0.
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[-1])
        return (d > self.target_radius - self.cursor_radius) and self.target_index == 0
    
    def _test_trial_incomplete(self, ts):
        return self.target_index < self.chain_length
    
    def _test_incorrect_target_penalty_end(self, ts):
        return ts > self.incorrect_target_penalty_time
    
    def _start_wait(self):
        super()._start_wait()
        # Redefine chain length because targs in this task has both eye and hand targets
        self.chain_length = len(self.targets)
        self.fixation_passed = False

        if self.calc_trial_num() == 0:

            # Instantiate the targets here so they don't show up in any states that might come before "wait"
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

            for target in self.targets_hand:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

    def _start_init_target(self):
        # Only show the hand target
        if self.target_index == -1:
            target_hand = self.targets_hand[0]
            target_hand.move_to_position(self.targs[-1])
            target_hand.show()
            self.sync_event('TARGET_ON', self.gen_indices[-1]) # the hand target is on  

        elif self.target_index == 0: # this is from the target state
            target = self.targets[self.target_index]
            target.hide()
            self.sync_event('EYE_TARGET_OFF', self.gen_indices[self.target_index])

    def _start_target(self):
        if self.target_index == -1 and not self.fixation_passed:
            self.target_index += 1

        if self.fixation_passed:
            self.target_index += 1

        # Show the eye target
        if self.target_index == 0:
            target = self.targets[self.target_index]
            target.move_to_position(self.targs[self.target_index] - self.offset_cube)
            target.show()
            self.sync_event('EYE_TARGET_ON', self.gen_indices[self.target_index])

    def _start_target_eye(self):
        self.target_index += 1

    def _start_fixation(self):
        self.fixation_passed = True
        self.targets[self.target_index].cube.color = target_colors[self.fixation_target_color] # change target color in fixation state
        self.sync_event('FIXATION', self.gen_indices[self.target_index])

    def _start_delay(self):
        # Make next target visible unless this is the final target in the trial
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length:
            target = self.targets[next_idx]
            target.move_to_position(self.targs[next_idx] - self.offset_cube)
            target.show()
            self.sync_event('EYE_TARGET_ON', self.gen_indices[next_idx])

    def _start_targ_transition(self):
        if self.target_index + 1 < self.chain_length:

            # Hide the current target if there are more
            self.targets[self.target_index].hide()
            self.sync_event('EYE_TARGET_OFF', self.gen_indices[self.target_index])

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        for target in self.targets_hand:
            target.hide()
            target.reset()
            
    def _start_hold_penalty(self):
        super()._start_hold_penalty()
        # Hide targets
        for target in self.targets_hand:
            target.hide()
            target.reset()

    def _start_delay_penalty(self):
        super()._start_delay_penalty()
        # Hide targets
        for target in self.targets_hand:
            target.hide()
            target.reset()

    def _start_fixation_penalty(self):
        self._increment_tries()
        self.sync_event('FIXATION_PENALTY') 
        self.penalty_index = 1

        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

        for target in self.targets_hand:
            target.hide()
            target.reset()

    def _end_fixation_penalty(self):
        self.sync_event('TRIAL_END')

    def _start_incorrect_target_penalty(self):
        self._increment_tries()
        self.sync_event('OTHER_PENALTY')
        self.penalty_index = 1

        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

        for target in self.targets_hand:
            target.cue_trial_end_failure()
            target.show()

    def _end_incorrect_target_penalty(self):
        self.sync_event('TRIAL_END')

        for target in self.targets_hand:
            target.hide()
            target.reset()

    def _start_reward(self):
        super()._start_reward()
        for target in self.targets_hand:
            target.cue_trial_end_success()

    def _end_reward(self):
        super()._end_reward()

        # Hide targets
        for target in self.targets_hand:
            target.hide()
            target.reset()        

    def _start_pause(self):
        super()._start_pause()

        # Hide targets
        for target in self.targets_hand:
            target.hide()
            target.reset()

    # Generator functions
    @staticmethod
    def row_target(nblocks=20, ntargets=3, dx=5.,offset1=(0,0,-2),offset2=(0,0,6.),offset3=(0,0,-7.5),origin=(0,0,0)):
        '''
        Generates a sequence of 3D for 2 eye targets and 1 hand target at a given distance from the origin

        Parameters
        ----------
        nblocks : int
            The number of ntarget pairs in the sequence.
        ntargets : int
            The number of equally spaced targets
        distance : float
            The distance in cm between targets
        offset1 : 3-tuple
            y location of the first eye target
        offset2 : 3-tuple
            y location of the second eye target
        offset3 : 3-tuple
            y location of the hand target
        origin : 3-tuple
            Location of the central targets

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [1 x 3] target coordinates

        '''
        rng = np.random.default_rng()
        for _ in range(nblocks):
            order = np.arange(ntargets**3)
            rng.shuffle(order)
            x_pos_candidate = [-dx,0,dx]
            for t in range(ntargets**3):
                idx = np.base_repr(order[t],3).zfill(3) # convert a decimal number to ternary

                # Target index for hand target, initial eye target, final eye target
                idx1 = int(idx[0])
                idx2 = int(idx[1])
                idx3 = int(idx[2])

                # Get positions for each target
                x_pos1 = x_pos_candidate[idx1]
                x_pos2 = x_pos_candidate[idx2]
                x_pos3 = x_pos_candidate[idx3]
                pos1 = np.array([x_pos1,0,0]).T
                pos2 = np.array([x_pos2,0,0]).T
                pos3 = np.array([x_pos3,0,0]).T

                yield [idx1],[idx2],[idx3],[pos1+offset1+origin],[pos2+offset2+origin],[pos3+offset3+origin]

    @staticmethod
    def sac_hand_2d(nblocks=20, ntargets=3, dx=10,offset1=(0,0,-2),offset2=(0,0,6.),offset3=(0,0,-7.5),origin=(0,0,0)):
        '''
        Pairs of hand targets and eye targets

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [3 x 3] target coordinates
        '''

        gen = HandConstrainedEyeCapture.row_target(nblocks=nblocks,ntargets=ntargets,dx=dx,offset1=offset1,offset2=offset2,offset3=offset3,origin=origin)
        for _ in range(nblocks*(ntargets**3)):
            idx1,idx2,idx3,pos1,pos2,pos3 = next(gen)

            targs = np.zeros([3, 3])
            targs[0,:] = pos1[0]
            targs[1,:] = pos2[0]
            targs[2,:] = pos3[0]

            indices = np.zeros([3,1])
            indices[0] = idx1[0]
            indices[1] = idx2[0] + ntargets
            indices[2] = idx3[0]

            yield indices, targs

class EyeConstrainedHandCapture(HandConstrainedEyeCapture):
    '''
    Saccade and reaching task. Subjects need to hold an initial hand target and fixate the initial eye target, separately. 
    Then the shared goal target appears for eye and hand. Subjects need to make a saccade and reach the goal target.
    The acceprance radius for eye fixation is fixation_radius + fixation_radius_buffer. The buffer radius is invisible for subjects. 
    2 of chain_length is only tested.
    '''

    status = dict(
        wait = dict(start_trial="init_target", start_pause="pause"),
        init_target = dict(enter_target="target", start_pause="pause"),
        target = dict(start_pause="pause", timeout="timeout_penalty", return_init_target='init_target', gaze_enter_target="fixation", gaze_incorrect_target="incorrect_target_penalty"),
        fixation = dict(start_pause="pause", leave_target="hold_penalty", fixation_hold_complete="delay", fixation_break="fixation_penalty"), 
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition", fixation_break="fixation_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        incorrect_target_penalty = dict(incorrect_target_penalty_end="wait", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )
 
    sequence_generators = ['row_target','sac_hand_2d']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:

            # Target 1 and 2 are for saccade. Target 3 and target 4 are for hand
            target1 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            target2 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            target3 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
            target4 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])

            self.targets = [target1, target2]
            self.targets_hand = [target3, target4]

            self.offset_cube = np.array([0,10,self.fixation_radius/2]) # To center the cube target

    def _test_gaze_enter_target(self,ts):
        '''
        Check whether eye positions and hand cursor are within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])

        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.targs[-1-self.target_index]) #targs[-1] is the first hand target, targ[-2] is the second target
        
        return (eye_d <= self.fixation_radius + self.fixation_radius_buffer) and (hand_d <= self.target_radius - self.cursor_radius)

    def _test_gaze_incorrect_target(self, ts):
        '''
        Check whether eye position is within the different target (hand target). This is only applied to the second target
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[-1,[0,2]])

        return eye_d <= (self.target_radius + self.incorrect_target_radius_buffer) and self.target_index == 1
    
    def _test_leave_target(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[-1-self.target_index])
        return d > self.target_radius - self.cursor_radius
    
    def _test_fixation_hold_complete(self,ts):
        return ts > self.fixation_time
    
    def _while_target(self):
        target = self.targets[self.target_index]
        target.move_to_position(self.targs[self.target_index] - self.offset_cube)

        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])
        
        if eye_d <= self.fixation_radius + self.fixation_radius_buffer:
            self.targets[self.target_index].cube.color = target_colors[self.fixation_target_color] # change target color in fixation state
        else:
            self.targets[self.target_index].cube.color = target_colors[self.eye_target_color]

    def _start_delay(self):
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length:
            # Show hand target
            target = self.targets_hand[next_idx]
            target.move_to_position(self.targs[next_idx])
            target.show() # Don't have to sync event because the second target is shared between hand and eye.

            # Show eye target
            target = self.targets[next_idx]
            target.move_to_position(self.targs[next_idx] - self.offset_cube)
            target.show()
            self.sync_event('EYE_TARGET_ON', self.gen_indices[next_idx])

    def _start_targ_transition(self):
        super()._start_targ_transition()
        if self.target_index + 1 < self.chain_length:

            # Hide the current hand target
            self.targets_hand[self.target_index].hide()

class EyeHandCaptureBlock(Sequence, Window):
    '''
    Subjects have to gaze at and reach to a target, responding to the eye or hand go cue indivisually in sequence trials.
    They need to keep their eye position while moving the cursor to the hand target.
    In simultaneous trials, they need to simultaneously move eye and hand to the target, responding a single go cue.
    '''

    eye_first = traits.Bool(True, desc="If True, eye_trials come first. If False, eye_hand_trials come first")
    trials_block_eye = traits.Int(108, desc="Trial numbers of the block in eye trials")
    trials_block_eye_hand = traits.Int(216, desc="Trial numbers of the block in eye-hand trials")
    reward_time_eye = traits.Float(.5, desc="Reward time in eye trials")
    reward_time_eye_hand = traits.Float(.5, desc="Reward time in eye-hand trials")
    trials_jackpot_eye = traits.Int(0, desc="Trial numbers to get jackpot in eye trials. If 0, no jackpot reward")
    trials_jackpot_eye_hand = traits.Int(27, desc="Trial numbers to get jackpot in eye-hand trials. If 0, no jackpot reward")
    reward_time_jackpot_eye = traits.Float(1.0, desc="Reward time for jackpot trials in eye trials")
    reward_time_jackpot_eye_hand = traits.Float(1.0, desc="Reward time for jackpot trials in eye-hand trials")

    fixation_time = traits.Float(.3, desc="fixation duration during which subjects have to keep fixating the first eye target")
    fixation_radius = traits.Float(2.5, desc="Width of the square eye target")
    fixation_radius_buffer = traits.Float(.5, desc="additional radius for eye target. fixation_radius (width) + buffer determines the break of fixation")
    fixation_penalty_time = traits.Float(1, desc="Length of penalty time for fixation break")
    hold_time = traits.Float(.2, desc="Length of hold required at targets before next target appears")
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    rand_delay_eye = traits.Tuple((0.4, 0.7), desc="Delay interval in eye trials")
    rand_delay_eye_hand = traits.Tuple((0.4, 0.7), desc="Delay interval in eye-hand trials")
    delay_penalty_time = traits.Float(1, desc="Length of penalty time for delay error")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")
    timeout_penalty_time = traits.Float(1, desc="Length of penalty time for timeout error")
    incorrect_target_radius_buffer = traits.Float(.5, desc="target radius + buffer radius determines if subjects look at the incorrect target")
    incorrect_target_penalty_time = traits.Float(1, desc="Length of penalty time for acquiring an incorrect target")
    max_attempts = traits.Int(10, desc="The number of attempts of a target chain before skipping to the next one")
    num_targets_per_attempt = traits.Int(2, desc="Minimum number of target acquisitions to be counted as an attempt")

    target_radius = traits.Float(2, desc="Radius of targets in cm")
    target_color = traits.OptionsList("yellow", *target_colors, desc="Color of the target", bmi3d_input_options=list(target_colors.keys()))
    plant_hide_rate = traits.Float(0.0, desc='If the plant is visible, specifies a percentage of trials where it will be hidden')
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))
    plant_visible = traits.Bool(True, desc='Specifies whether entire plant is displayed or just endpoint')
    cursor_radius = traits.Float(.5, desc='Radius of cursor in cm')
    cursor_color = traits.OptionsList("dark_purple", *target_colors, desc='Color of cursor endpoint', bmi3d_input_options=list(target_colors.keys()))
    cursor_bounds = traits.Tuple((-10., 10., -10., 10., -10., 10.), desc='(x min, x max, y min, y max, z min, z max)')
    starting_pos = traits.Tuple((5., 0., 5.), desc='Where to initialize the cursor') 

    init_eye_target_alpha = traits.Float(1., desc="Transparency of initial eye targets")
    init_hand_target_alpha = traits.Float(0.9, desc="Transparency of initial hand targets")
    goal_eye_target_alpha = traits.Float(0.75, desc="Transparency of goal eye targets")
    goal_hand_target_alpha = traits.Float(0.75, desc="Transparency of goal hand targets")

    fixation_target_color = traits.OptionsList("fixation_color", *target_colors, desc="Color of the eye target under fixation state", bmi3d_input_options=list(target_colors.keys()))
    eye_target_color = traits.OptionsList("eye_color", *target_colors, desc="Color of the eye target", bmi3d_input_options=list(target_colors.keys()))
    limit2d = traits.Bool(True, desc="Limit cursor movement to 2D")
    hidden_traits = ['eye_target_color', 'fixation_target_color','cursor_color', \
                     'target_color', 'cursor_bounds', 'cursor_radius', 'plant_hide_rate', 'starting_pos']

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(enter_target="target_eye", start_pause="pause"),
        target_eye = dict(timeout="timeout_penalty", return_init_target='target', leave_target="hold_penalty", gaze_target="fixation", \
                        gaze_incorrect_target="incorrect_target_penalty", start_pause="pause"),
        target_eye_hand = dict(timeout="timeout_penalty", gaze_enter_target='hold', \
                        gaze_incorrect_target="incorrect_target_penalty", start_pause="pause"),
        fixation = dict(fixation_complete="delay", leave_target="hold_penalty", fixation_break="fixation_penalty", start_pause="pause"),
        hold = dict(hold_complete="delay", leave_target="hold_penalty",  fixation_break="fixation_penalty", start_pause="pause"),
        delay = dict(delay_complete="targ_transition", leave_target="delay_penalty", fixation_break="delay_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", targ_eye_hand="target_eye_hand", targ_eye="target_eye", start_pause="pause"),
        incorrect_target_penalty = dict(incorrect_target_penalty_end="wait", start_pause="pause", end_state=True),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    # initial state
    state = "wait"
    sequence_generators = ['row_target','sac_hand_2d']

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

            # Control transparency of targets
            init_eye_new_color = list(target_colors[self.eye_target_color])
            init_eye_new_color[3] = self.init_eye_target_alpha
            init_hand_new_color = list(target_colors[self.target_color])
            init_hand_new_color[3] = self.init_hand_target_alpha
            goal_eye_new_color = list(target_colors[self.eye_target_color])
            goal_eye_new_color[3] = self.goal_eye_target_alpha
            goal_hand_new_color = list(target_colors[self.target_color])
            goal_hand_new_color[3] = self.goal_hand_target_alpha

            # Target 1 and 2 are for saccade. Target 3 and target 4 are for hand
            target1 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=init_eye_new_color)
            target2 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=goal_eye_new_color)
            target3 = VirtualCircularTarget(target_radius=self.target_radius, target_color=init_hand_new_color)
            target4 = VirtualCircularTarget(target_radius=self.target_radius, target_color=goal_hand_new_color)

            self.targets = [target1, target2]
            self.targets_hand = [target3, target4]

            self.offset_cube = np.array([0,10,self.fixation_radius/2]) # To center the cube target
        self.target_location = np.array(self.starting_pos).copy()

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)

        # Initialize these values for report_stats
        self.trials_all_blocks = self.trials_block_eye + self.trials_block_eye_hand
        self.trial_count_blocks = self.reward_count % self.trials_all_blocks

        self.reward_count_eye = 0
        self.reward_count_eye_hand = 0

        if self.eye_first:
            self.is_eye_trials = True
            self.is_eye_hand_trials = False
        else:
            self.is_eye_trials = False
            self.is_eye_hand_trials = True           

    def init(self):
        self.trial_dtype = np.dtype([('trial', 'u4'), ('index', 'u4'), ('target', 'f8', (3,))])
        self.add_dtype('is_eye_trials', bool, (1,))
        self.add_dtype('trial', 'u4', (1,))
        self.add_dtype('plant_visible', '?', (1,))
        self.penalty_index = 0
        self.pause_index = 0
        self.total_pause_time = 0
        self.current_pause_time = 0
        super().init()
        self.plant.set_endpoint_pos(np.array(self.starting_pos))

    def _cycle(self):
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

    def move_effector(self, *args, **kwargs):
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

    def update_plant_visibility(self):
        ''' Update plant visibility'''
        if self.plant_visible != self.plant_vis_prev:
            self.plant_vis_prev = self.plant_visible
            self.plant.set_visibility(self.plant_visible)

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

    def _parse_next_trial(self):
        '''Check that the generator has the required data'''
        # 2 target positions for hand and eye. The first and second target index is for eye, and the third one is for hand
        self.gen_indices, self.targs = self.next_trial 

        # Update the data sinks with trial information
        self.trial_record['trial'] = self.calc_trial_num()
        for i in range(len(self.gen_indices)):
            self.trial_record['index'] = self.gen_indices[i]
            self.trial_record['target'] = self.targs[i]
            self.sinks.send("trials", self.trial_record)

    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super().update_report_stats()
        self.reportstats['Trial #'] = self.calc_trial_num()
        self.reportstats['Reward/min'] = np.round(self.calc_events_per_min('reward', 120.), decimals=2)
        self.reportstats['Total pause time'] = self._time_to_string(self.total_pause_time)
        self.reportstats['Current pause time'] = self._time_to_string(self.current_pause_time)

        self.trial_count_blocks = self._fast_state_counts['reward'] % self.trials_all_blocks
        if self.is_eye_trials:
            self.reportstats['Task of this block'] = 'Eye'
            current_count = self.trial_count_blocks if self.eye_first else self.trial_count_blocks - self.trials_block_eye_hand
            self.reportstats['Success trial # / Block'] = f'{current_count} / {self.trials_block_eye}'
        else:
            self.reportstats['Task of this block'] = 'Eye-Hand'
            current_count = self.trial_count_blocks - self.trials_block_eye if self.eye_first else self.trial_count_blocks
            self.reportstats['Success trial # / Block'] = f'{current_count} / {self.trials_block_eye_hand}'

        self.reportstats['Success trial # (eye trials)'] = self.reward_count_eye
        self.reportstats['Success trial # (eye-hand trials)'] = self.reward_count_eye_hand

    def _start_wait(self):
        self.fixation_passed = False
        self.chain_length = len(self.targets)
        self.plant.cursor.attach()

        if self.calc_trial_num() == 0:

            # Instantiate the targets here so they don't show up in any states that might come before "wait"
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

            for target in self.targets_hand:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

        if self.penalty_index == 0 and self.pause_index == 0: # doesn't call parent method when the state comes from the penalty or pause state
            # Call parent method to draw the next target capture sequence from the generator
            super()._start_wait()
            self.tries = 0 # number of times this sequence of targets has been attempted

        if self.tries==self.max_attempts: # The task goes to the next target after the number of reattempting is max attempts 
            super()._start_wait()
            self.tries = 0 # number of times this sequence of targets has been attempted

        if self.tries == 0: # Update delay_time only in the first attempt
            
            # Decide eye or eye-hand trials  
            self.trial_count_blocks = self._fast_state_counts['reward'] % self.trials_all_blocks

            if self.eye_first:
                # Trial blocks changes from eye to eye-hand
                if self.trial_count_blocks < self.trials_block_eye:
                    self._set_trial_type(is_eye=True)
                else:
                    self._set_trial_type(is_eye=False)
            else:
                # Trial blocks changes from eye-hand to eye
                if self.trial_count_blocks < self.trials_block_eye_hand:
                    self._set_trial_type(is_eye=False)
                else:
                    self._set_trial_type(is_eye=True)

            self.task_data['is_eye_trials'] = self.is_eye_trials

        # index of current target presented to subject
        self.target_index = -1

        # Set index to 0 because the state may come from the penalty or pause state,
        self.penalty_index = 0
        self.pause_index = 0

    def _set_trial_type(self, is_eye):
        if is_eye:
            self.is_eye_trials = True
            self.is_eye_hand_trials = False

            s, e = self.rand_delay_eye
            self.delay_time_eye = random.random()*(e-s) + s

            self.reward_time = self.reward_time_eye
            if self.reward_count_eye % self.trials_jackpot_eye == self.trials_jackpot_eye-1:
                self.reward_time = self.reward_time_jackpot_eye

        else:
            self.is_eye_trials = False
            self.is_eye_hand_trials = True
            
            s, e = self.rand_delay_eye_hand
            self.delay_time_eye_hand = random.random()*(e-s) + s

            self.reward_time = self.reward_time_eye_hand
            if self.reward_count_eye_hand % self.trials_jackpot_eye_hand == self.trials_jackpot_eye_hand-1:
                self.reward_time = self.reward_time_jackpot_eye_hand

    def _start_target(self):
        # Only show the hand target
        if self.target_index == -1:
            target_hand = self.targets_hand[0]
            target_hand.move_to_position(self.targs[-1])
            target_hand.show()
            self.sync_event('TARGET_ON', self.gen_indices[-1]) # the hand target is on  

        elif self.target_index == 0: # this is from target_eye state
            target = self.targets[self.target_index]
            target.hide()
            self.sync_event('EYE_TARGET_OFF', self.gen_indices[self.target_index])

    def _start_target_eye(self):
        if self.target_index == -1 and not self.fixation_passed:
            self.target_index += 1

        if self.fixation_passed:
            self.target_index += 1

        # Show the eye target
        if self.target_index == 0:
            target = self.targets[self.target_index]
            target.move_to_position(self.targs[self.target_index] - self.offset_cube)
            target.show()
            self.sync_event('EYE_TARGET_ON', self.gen_indices[self.target_index])

    def _start_target_eye_hand(self):
        if self.target_index == -1 and not self.fixation_passed:
            self.target_index += 1

        if self.fixation_passed:
            self.target_index += 1
        
    def _while_target_eye_hand(self):
        target = self.targets[self.target_index]
        target.move_to_position(self.targs[self.target_index] - self.offset_cube)

        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])
        
        if eye_d <= self.fixation_radius + self.fixation_radius_buffer:
            self.targets[self.target_index].cube.color = target_colors[self.fixation_target_color] # change target color in fixation state
        else:
            self.targets[self.target_index].cube.color = target_colors[self.eye_target_color]

    def _start_fixation(self):
        self.fixation_passed = True
        self.targets[self.target_index].cube.color = target_colors[self.fixation_target_color] # change target color in fixation state
        self.sync_event('FIXATION', self.gen_indices[self.target_index])

    def _start_hold(self):
        self.sync_event('CURSOR_ENTER_TARGET', self.gen_indices[self.target_index])
        
    def _start_delay(self):
        # Make next target visible unless this is the final target in the trial
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length:
            # Show hand target
            if self.is_eye_hand_trials:
                target = self.targets_hand[next_idx]
                target.move_to_position(self.targs[next_idx])
                target.show() # Don't have to sync event because the second target is shared between hand and eye.

            # Show eye target
            target = self.targets[next_idx]
            target.move_to_position(self.targs[next_idx] - self.offset_cube)
            target.show()
            self.sync_event('EYE_TARGET_ON', self.gen_indices[next_idx])

    def _start_targ_transition(self):
        if self.target_index + 1 < self.chain_length:

            # Hide the current target if there are more
            self.targets[self.target_index].hide()
            self.sync_event('EYE_TARGET_OFF', self.gen_indices[self.target_index])
            
            if self.is_eye_hand_trials:
                self.targets_hand[self.target_index].hide()

    def _start_timeout_penalty(self):
        self._increment_tries()
        self.penalty_index = 1
        self.sync_event('TIMEOUT_PENALTY')

        for target in self.targets:
            target.hide()
            target.reset()

        for target in self.targets_hand:
            target.hide()
            target.reset()

    def _end_timeout_penalty(self):
        self.sync_event('TRIAL_END')

    def _start_hold_penalty(self):
        self._increment_tries()
        self.penalty_index = 1
        self.sync_event('HOLD_PENALTY')

        for target in self.targets:
            target.hide()
            target.reset()

        for target in self.targets_hand:
            target.hide()
            target.reset()

    def _end_hold_penalty(self):
        self.sync_event('TRIAL_END')

    def _start_delay_penalty(self):
        self._increment_tries()
        self.penalty_index = 1        
        self.sync_event('DELAY_PENALTY') 

        for target in self.targets:
            target.hide()
            target.reset()

        for target in self.targets_hand:
            target.hide()
            target.reset()

    def _end_delay_penalty(self):
        self.sync_event('TRIAL_END')

    def _start_fixation_penalty(self):
        self._increment_tries()
        self.penalty_index = 1
        self.sync_event('FIXATION_PENALTY') 

        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

        for target in self.targets_hand:
            target.hide()
            target.reset()

    def _end_fixation_penalty(self):
        self.sync_event('TRIAL_END')

    def _start_incorrect_target_penalty(self):
        self._increment_tries()
        self.sync_event('OTHER_PENALTY')
        self.penalty_index = 1

        self.targets[1].cue_trial_end_failure()
        self.targets[1].show()

        self.targets_hand[0].cue_trial_end_failure()
        self.targets_hand[0].show()

        if self.is_eye_hand_trials:
            self.targets_hand[1].cue_trial_end_failure()
            self.targets_hand[1].show()            

    def _end_incorrect_target_penalty(self):
        self.sync_event('TRIAL_END')

        for target in self.targets:
            target.hide()
            target.reset()

        for target in self.targets_hand:
            target.hide()
            target.reset()     

    def _start_reward(self):
        self.targets[self.target_index].cue_trial_end_success()
        self.targets_hand[self.target_index].cue_trial_end_success()
        self.targets_hand[0].hide()
        self.plant.cursor.detach()
        self.sync_event('REWARD')

        if self.is_eye_trials:
            self.reward_count_eye += 1
        else:
            self.reward_count_eye_hand += 1

    def _end_reward(self):
        self.sync_event('TRIAL_END')

        for target in self.targets:
            target.hide()
            target.reset()

        for target in self.targets_hand:
            target.hide()
            target.reset()        

    def _start_pause(self):
        self.pause_index = 1
        self.sync_event('PAUSE_START')

        self.pause_start_time = self.get_time()
        self.total_pause_time_old = self.total_pause_time

        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

        for target in self.targets_hand:
            target.hide()
            target.reset()    

    def _while_pause(self):
        self.current_pause_time = self.get_time() - self.pause_start_time
        self.total_pause_time = self.total_pause_time_old + self.current_pause_time

    def _end_pause(self):
        self.sync_event('PAUSE_END')
        self.current_pause_time = 0

    def _test_start_trial(self, time_in_state):
        return True
    
    def _test_gaze_enter_target(self,ts):
        '''
        Check whether eye positions and hand cursor are within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])

        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.targs[-1-self.target_index]) #targs[-1] is the first hand target, targ[-2] is the second target
        
        return (eye_d <= self.fixation_radius + self.fixation_radius_buffer) and (hand_d <= self.target_radius - self.cursor_radius)
    
    def _test_gaze_target(self, ts):
        '''
        Check whether eye position is within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])

        return eye_d <= self.fixation_radius + self.fixation_radius_buffer   
    
    def _test_gaze_incorrect_target(self, ts):
        '''
        Check whether eye position is within the different target (hand target) only when target_index == 1
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[-1,[0,2]])

        return eye_d <= self.target_radius + self.incorrect_target_radius_buffer and self.target_index == 1
    
    def _test_fixation_break(self,ts):
        '''
        Triggers the fixation_penalty state when eye positions are outside fixation distance
        '''
        # Distance of an eye position from a target position
        eye_pos = self.calibrated_eye_pos
        d_eye = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])
        return d_eye > self.fixation_radius + self.fixation_radius_buffer
    
    def _test_fixation_complete(self,ts):
        return ts > self.fixation_time
    
    def _test_hold_complete(self, time_in_state):
        return time_in_state > self.hold_time
    
    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[-1]) # hand must be within the initial target
        return d <= self.target_radius - self.cursor_radius

    def _test_leave_target(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        if self.is_eye_hand_trials:
            d = np.linalg.norm(cursor_pos - self.targs[-1-self.target_index])
        else:
            d = np.linalg.norm(cursor_pos - self.targs[-1])
        return d > self.target_radius - self.cursor_radius

    def _test_return_init_target(self, ts):
        '''
        return true if cursor moves outside the exit radius, but only applied when the target index is 0.
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[-1])
        return (d > self.target_radius - self.cursor_radius) and self.target_index == 0

    def _test_delay_complete(self, ts):
        '''
        Test whether the delay period is over. In sequence trials, there are 2 delay period for each eye and hand.
        In simultaneous trials, there is only 1 delay period.
        '''
        if self.target_index == 0 and self.is_eye_trials:
            return ts > self.delay_time_eye
        elif self.target_index == 0 and self.is_eye_hand_trials:
            return ts > self.delay_time_eye_hand
        else:
            return True

    def _test_targ_eye(self,ts):
        return self.target_index == 0 and self.is_eye_trials
    
    def _test_targ_eye_hand(self,ts):
        return self.target_index == 0 and self.is_eye_hand_trials

    def _test_trial_complete(self, time_in_state):
        return self.target_index == self.chain_length-1
    
    def _test_trial_abort(self, time_in_state):
        return (not self._test_trial_complete(time_in_state)) and (self.tries==self.max_attempts)

    def _test_trial_incomplete(self, time_in_state):
        return (not self._test_trial_complete(time_in_state)) and (self.tries<self.max_attempts)

    def _test_timeout(self, time_in_state):
        return time_in_state > self.timeout_time
    
    def _test_timeout_penalty_end(self, time_in_state):
        return time_in_state > self.timeout_penalty_time

    def _test_hold_penalty_end(self, time_in_state):
        return time_in_state > self.hold_penalty_time

    def _test_delay_penalty_end(self, time_in_state):
        return time_in_state > self.delay_penalty_time

    def _test_fixation_penalty_end(self,ts):
        return ts > self.fixation_penalty_time

    def _test_incorrect_target_penalty_end(self, ts):
        return ts > self.incorrect_target_penalty_time
    
    def _test_reward_end(self, time_in_state):
        return time_in_state > self.reward_time

    def _test_start_pause(self, time_in_state):
        return self.pause

    def _test_end_pause(self, time_in_state):
        return not self.pause

    # Generator functions
    @staticmethod
    def row_target(nblocks=20, ntargets=3, dx=5.,offset1=(0,0,-2),offset2=(0,0,6.),offset3=(0,0,-7.5),origin=(0,0,0)):
        '''
        Generates a sequence of 3D for 2 eye targets and 1 hand target at a given distance from the origin

        Parameters
        ----------
        nblocks : int
            The number of ntarget pairs in the sequence.
        ntargets : int
            The number of equally spaced targets
        distance : float
            The distance in cm between targets
        offset1 : 3-tuple
            y location of the first eye target
        offset2 : 3-tuple
            y location of the second eye target
        offset3 : 3-tuple
            y location of the hand target
        origin : 3-tuple
            Location of the central targets

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [1 x 3] target coordinates

        '''
        rng = np.random.default_rng()
        for _ in range(nblocks):
            order = np.arange(ntargets**3)
            rng.shuffle(order)
            x_pos_candidate = [-dx,0,dx]
            for t in range(ntargets**3):
                idx = np.base_repr(order[t],3).zfill(3) # convert a decimal number to ternary

                # Target index for hand target, initial eye target, final eye target
                idx1 = int(idx[0])
                idx2 = int(idx[1])
                idx3 = int(idx[2])

                # Get positions for each target
                x_pos1 = x_pos_candidate[idx1]
                x_pos2 = x_pos_candidate[idx2]
                x_pos3 = x_pos_candidate[idx3]
                pos1 = np.array([x_pos1,0,0]).T
                pos2 = np.array([x_pos2,0,0]).T
                pos3 = np.array([x_pos3,0,0]).T

                yield [idx1],[idx2],[idx3],[pos1+offset1+origin],[pos2+offset2+origin],[pos3+offset3+origin]

    @staticmethod
    def sac_hand_2d(nblocks=20, ntargets=3, dx=10,offset1=(0,0,-2),offset2=(0,0,6.),offset3=(0,0,-7.5),origin=(0,0,0)):
        '''
        Pairs of hand targets and eye targets

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [3 x 3] target coordinates
        '''

        gen = HandConstrainedEyeCapture.row_target(nblocks=nblocks,ntargets=ntargets,dx=dx,offset1=offset1,offset2=offset2,offset3=offset3,origin=origin)
        for _ in range(nblocks*(ntargets**3)):
            idx1,idx2,idx3,pos1,pos2,pos3 = next(gen)

            targs = np.zeros([3, 3])
            targs[0,:] = pos1[0]
            targs[1,:] = pos2[0]
            targs[2,:] = pos3[0]

            indices = np.zeros([3,1])
            indices[0] = idx1[0]
            indices[1] = idx2[0] + ntargets
            indices[2] = idx3[0]

            yield indices, targs

class EyeHandCaptureBlock_sequence(EyeHandCaptureBlock):
    '''
    This task is similar to the EyeHandCaptureBlock. In eye trials, the task structure is exactly the same as the EyeHandCaptureBlock.
    For eye-hand trials, subjects need to move their eyes first, then move their arm while keeping their fixation on the eye target.
    '''

    trials_online_eye_calib = traits.Int(0, desc='Trial numbers used for online eye calibration. If 0, no online calibration is performed')
    offset_time_online_eye_calib = traits.Float(0.1, desc="Data after this offset_time is only used for eye calibration in the fixation state")
    duration_online_eye_calib = traits.Float(0.1, desc="Data within this duration after offset_time in the fixation state is only used for eye calibration")
    fixation_time1 = traits.Float(.2, desc="First fixation duration. This is both for eye_trials nad eye_hand_trials")
    fixation_time2 = traits.Float(.2, desc="Second fixation duration. This is only for eye_hand_trials")
    exclude_parent_traits = ['fixation_time']

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(enter_target="target_eye", start_pause="pause"),
        target_eye = dict(timeout="timeout_penalty", return_init_target='target', leave_target="hold_penalty", gaze_target="fixation", \
                        gaze_incorrect_target="incorrect_target_penalty", start_pause="pause"),
        fixation = dict(fixation_complete="target_hand", leave_target="hold_penalty", fixation_break="fixation_penalty", start_pause="pause"),
        target_hand = dict(timeout="timeout_penalty", pass_target_hand='delay', enter_target='hold', \
                        fixation_break="fixation_penalty", start_pause="pause"),
        hold = dict(hold_complete="delay", leave_target="hold_penalty",  fixation_break="fixation_penalty", start_pause="pause"),
        delay = dict(delay_complete="targ_transition", leave_target="delay_penalty", fixation_break="delay_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", targ_eye="target_eye", start_pause="pause"),
        incorrect_target_penalty = dict(incorrect_target_penalty_end="wait", start_pause="pause", end_state=True),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.trials_online_eye_calib:
            self.eye_coeff_stack = []
            self.m_eye_pos_online_eye_calib = deque(maxlen = self.trials_online_eye_calib*2) # bacause there are 2 eye pos within a single trial
            self.targ_pos_online_eye_calib = deque(maxlen = self.trials_online_eye_calib*2)

    def _start_wait(self):
        super()._start_wait()
        self.hand_target_index = -1

        if self.trials_online_eye_calib:
            self.eye_pos_tmp0 = []
            self.eye_pos_tmp1 = []

            if self.tries == 0:
                if self.calc_state_occurrences('reward') > self.trials_online_eye_calib:
                    if len(self.eye_pos) == 4:
                        target_pos_tile = np.tile(np.array(self.targ_pos_online_eye_calib), (1,2))
                    else:
                        target_pos_tile = np.array(self.targ_pos_online_eye_calib)

                    slopes, intercepts, _ = aopy.analysis.fit_linear_regression(np.array(self.m_eye_pos_online_eye_calib), target_pos_tile)
                    self.eye_coeff = np.vstack((slopes, intercepts)).T
                    self.eye_center = np.zeros((4,))

                    self.eye_coeff_stack.append(self.eye_coeff)

    def _start_fixation(self):
        super()._start_fixation()
        if self.trials_online_eye_calib:
            self.start_fixation_time = self.get_time()

    def _while_fixation(self):
        # Only store eye pos between offset ~ offset + duration in the fixation state of eye trials
        if self.trials_online_eye_calib:
            elapsed_time = self.get_time() - self.start_fixation_time
            if elapsed_time > self.offset_time_online_eye_calib and elapsed_time < self.offset_time_online_eye_calib + self.duration_online_eye_calib:
                if self.target_index == 0:
                    self.eye_pos_tmp0.append(self.eye_pos[:4]) # store data regardless of whther it is eye trials or eye-hand trials
                elif self.target_index == 1 and self.is_eye_trials:
                    self.eye_pos_tmp1.append(self.eye_pos[:4]) # store data only from eye trials

    def _start_hold(self):
        super()._start_hold()
        if self.trials_online_eye_calib:
            self.start_hold_time = self.get_time()

    def _while_hold(self):
        # Only store eye pos between offset ~ offset + duration in the hold state of eye-hand trials
        if self.trials_online_eye_calib:
            elapsed_time = self.get_time() - self.start_hold_time
            if elapsed_time > self.offset_time_online_eye_calib and elapsed_time < self.offset_time_online_eye_calib + self.duration_online_eye_calib:
                if self.hand_target_index == 1 and self.is_eye_hand_trials:
                    self.eye_pos_tmp1.append(self.eye_pos[:4]) # store data only from eye-hand trials

    def _start_target_hand(self):
        # Hide hand target to show the go cue for hand movement
        if self.is_eye_hand_trials and self.target_index == 1:
            self.hand_target_index = 1
            self.targets_hand[self.hand_target_index-1].hide()
            self.sync_event('TARGET_OFF', self.gen_indices[self.hand_target_index-1])

    def _end_reward(self):
        super()._end_reward()
        if self.trials_online_eye_calib:
            self.m_eye_pos_online_eye_calib.append(np.nanmean(self.eye_pos_tmp0, axis=0))
            self.m_eye_pos_online_eye_calib.append(np.nanmean(self.eye_pos_tmp1, axis=0))
            self.targ_pos_online_eye_calib.append(np.array(self.targs)[0,[0,2]])
            self.targ_pos_online_eye_calib.append(np.array(self.targs)[1,[0,2]])

    def _start_targ_transition(self):
        if self.target_index + 1 < self.chain_length:

            # Hide the current target if there are more
            self.targets[self.target_index].hide()
            self.sync_event('EYE_TARGET_OFF', self.gen_indices[self.target_index])

    def _test_fixation_complete(self,ts):

        if self.is_eye_hand_trials and self.target_index == 1:
            return ts > self.fixation_time2
        else:
            return ts > self.fixation_time1

    
    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[self.hand_target_index])
        return d <= self.target_radius - self.cursor_radius
    
    def _test_leave_target(self, ts):
        '''
        return true if cursor moves outside the hand target radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[self.hand_target_index])
        return d > self.target_radius - self.cursor_radius
    
    def _test_pass_target_hand(self, ts):
        '''
        whether to pass target_hand state when trials are eye_trials and the target index is 0
        '''
        if self.is_eye_trials:
            return True
        elif self.target_index == 0:
            return True
        else:
            return False

    def _test_targ_eye(self,ts):
        return self.target_index < self.chain_length - 1

    def cleanup_hdf(self):
        super().cleanup_hdf()
        if self.trials_online_eye_calib:
            if hasattr(self, "h5file"):
                h5file = tables.open_file(self.h5file.name, mode='a')
                h5file.root.task.attrs['eye_coeff_stack'] = np.array(self.eye_coeff_stack)
                h5file.close()

class EyeHandSequenceCapture(EyeConstrainedTargetCapture):
    '''
    Subjects have to gaze at and reach to a target, responding to the eye or hand go cue indivisually in sequence trials.
    They need to keep their eye position while moving the cursor to the hand target.
    In simultaneous trials, they need to simultaneously move eye and hand to the target, responding a single go cue.
    '''

    exclude_parent_traits = ['delay_time', 'rand_delay','prob_catch_trials','short_delay_catch_trials','reward_time']

    trials_online_eye_calib = traits.Int(0, desc='Trial numbers used for online eye calibration. If 0, no online calibration is performed')
    offset_time_online_eye_calib = traits.Float(0.1, desc="Data after this offset_time is only used for eye calibration in the fixation state")
    duration_online_eye_calib = traits.Float(0.1, desc="Data within this duration after offset_time in the fixation state is only used for eye calibration")

    rand_delay_eye_hand = traits.Tuple((0.4, 0.7), desc="Delay interval for eye and hand in simultaneous trials")
    rand_delay_eye = traits.Tuple((0.4, 0.7), desc="Delay interval for eye in sequence trials")
    rand_delay_hand = traits.Tuple((0., 0.5), desc="Delay interval for hand in sequence trials")
    fixation_time = traits.Float(.3, desc="fixation duration during which subjects have to keep fixating the target")
    hold_time = traits.Float(.3, desc="hold duration during which subjects have to hold the second target in sequence trials")
    trials_block_simultaneous = traits.Int(100, desc='Trial numbers of the block in simultaneous trials')
    trials_block_sequence = traits.Int(100, desc='Trial numbers of the block in sequence trials')
    
    reward_time_simultaneous = traits.Float(.5, desc="Reward time in simultaneous trials")
    reward_time_sequence = traits.Float(.7, desc="Reward time in sequence trials")
    trials_jackpot_simultaneous = traits.Int(0, desc="Trial numbers to get jackpot in simultaneous trials. If 0, no jackpot reward")
    trials_jackpot_sequence = traits.Int(27, desc="Trial numbers to get jackpot in sequence trials. If 0, no jackpot reward")
    reward_time_jackpot_simultaneous = traits.Float(1.0, desc="Reward time for jackpot trials in simultaneous trials")
    reward_time_jackpot_sequence = traits.Float(1.0, desc="Reward time for jackpot trials in sequence trials")    

    diff_eye_hand_RTs_thr = traits.Float(0.5, desc="Accepted difference between eye and hand RTs in simultaneous trials")
    coordination_penalty_time = traits.Float(0.5, desc="Length of penalty time for less coordinated eye and hand movement in simultaneous trials")
    hand_RTs_thr_simul = traits.Float(0.55, desc="Accepted reach RTs in simultaneous trials")
    hand_RTs_thr_seq = traits.Float(0.55, desc="Accepted reach RTs in sequence trials")
    tooslow_penalty_time = traits.Float(0.5, desc="Length of penalty time for too slow reach RTs in both simultaneous and sequence trials")
    sequence_target_color = traits.OptionsList("orange", *target_colors, desc="Color of the hand target in sequence trials", bmi3d_input_options=list(target_colors.keys()))
    sequence_gocue_color = traits.OptionsList("pink", *target_colors, desc="Color of go cue in sequence trials", bmi3d_input_options=list(target_colors.keys()))
    hidden_traits = ['sequence_target_color', 'sequence_gocue_color']

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(timeout="timeout_penalty", gaze_enter_target='fixation', start_pause="pause"),
        target_eye = dict(timeout="timeout_penalty", gaze_target="fixation", leave_target="hold_penalty", start_pause="pause"),
        target_hand = dict(timeout="timeout_penalty", slow_reach_onset='tooslow_penalty', enter_target="hold", fixation_break="fixation_penalty", start_pause="pause"),
        target_eye_hand = dict(timeout="timeout_penalty", coordination_break='coordination_penalty', slow_reach_onset='tooslow_penalty', gaze_enter_target='fixation', start_pause="pause"),
        fixation = dict(fixation_complete="delay", leave_target="hold_penalty", fixation_break="fixation_penalty", start_pause="pause"),
        hold = dict(hold_complete="delay", leave_target="hold_penalty",  fixation_break="fixation_penalty", start_pause="pause"),
        delay = dict(delay_complete="targ_transition", leave_target="delay_penalty", fixation_break="fixation_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", targ_simultaneous="target_eye_hand",\
                               targ_first_sequence="target_eye", targ_second_sequence="target_hand", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        coordination_penalty = dict(coordination_penalty_end="wait", start_pause="pause", end_state=True),
        tooslow_penalty = dict(tooslow_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:

            # Target 1 and 2 are for eye targets. Target 3 and 4 is for hand targets
            target1 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            target2 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            target3 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
            target4 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])

            self.targets_eye = [target1, target2]
            self.targets_hand = [target3, target4]

            self.offset_cube = np.array([0,10,self.fixation_radius/2]) # To center the cube target

        # Initialize these values for report_stats
        self.trials_all_blocks = self.trials_block_sequence + self.trials_block_simultaneous
        self.trial_count_blocks = self.reward_count % self.trials_all_blocks
        self.is_sequence = False
        self.is_simultaneous = True

        self.reward_count_simultaneous = 0
        self.reward_count_sequence = 0

        # For eye calibration during the task
        if self.trials_online_eye_calib:
            self.eye_coeff_stack = []
            self.m_eye_pos_online_eye_calib = deque(maxlen = self.trials_online_eye_calib)
            self.m_eye_pos_center_online_eye_calib = deque(maxlen = self.trials_online_eye_calib)
            self.targ_pos_online_eye_calib = deque(maxlen = self.trials_online_eye_calib)

    def init(self):
        self.add_dtype('is_sequence', bool, (1,))
        super().init()

    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super().update_report_stats()
        self.trial_count_blocks = self.calc_state_occurrences('reward') % self.trials_all_blocks
        if self.is_simultaneous:
            self.reportstats['Task of this block'] = 'Joint'
            self.reportstats['Success trial # / Block'] = f'{self.trial_count_blocks} / {self.trials_block_simultaneous}'
        else:
            self.reportstats['Task of this block'] = 'Sequence'
            self.reportstats['Success trial # / Block'] = f'{self.trial_count_blocks - self.trials_block_simultaneous} / {self.trials_block_sequence}'

        self.reportstats['Success trial # (joint trials)'] = self.reward_count_simultaneous
        self.reportstats['Success trial # (sequence trials)'] = self.reward_count_sequence

    def _test_gaze_enter_target(self,ts):
        '''
        Check whether eye positions and hand cursor are within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])

        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.hand_targs[self.hand_target_index])
        
        return (eye_d <= self.fixation_radius + self.fixation_radius_buffer) and (hand_d <= self.target_radius - self.cursor_radius)

    def _test_gaze_target(self,ts):
        '''
        Check whether eye positions are within the fixation distance
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])
        return eye_d <= self.fixation_radius + self.fixation_radius_buffer

    def _test_leave_target(self, ts):
        '''
        check whether the hand cursor is outside the target distance
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.hand_targs[self.hand_target_index])
        return hand_d > (self.target_radius - self.cursor_radius)    
        
    def _test_enter_target(self, ts):
        '''
        check whether the hand cursor is within the target distance
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.hand_targs[self.hand_target_index])
        return hand_d <= (self.target_radius - self.cursor_radius)
    
    def _test_coordination_break(self,ts):
        '''
        check whether eye and hand reaction times are similar
        '''
        # Compute reaction time for eye
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[0,[0,2]]) # distance from the center target

        if eye_d > self.target_radius and self.reaction_time_eye == 0:
            self.reaction_time_eye = ts

        # Compute reaction time for hand
        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.hand_targs[0]) # distance from the center target

        if hand_d > (self.target_radius - self.cursor_radius) and self.reaction_time_hand == 0:
            self.reaction_time_hand = ts
        
        if self.reaction_time_hand == 0 or self.reaction_time_eye == 0:
            return False
        else:
            return np.abs(self.reaction_time_hand - self.reaction_time_eye) > self.diff_eye_hand_RTs_thr
    
    def _test_slow_reach_onset(self,ts):
        '''
        check whether the hand cursor is still within the center taret even when a ceratin amount of time (reaction_time_thr) passed
        '''
        # Compute reaction time for hand
        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.hand_targs[0]) # distance from the center target

        if ts > self.reaction_time_thr:
            return hand_d < (self.target_radius - self.cursor_radius)
        else:
            return False
    
    def _test_fixation_break(self,ts):
        '''
        Triggers the fixation_penalty state when eye positions are outside fixation distance
        Only apply this to the first hold and delay period
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])
        return eye_d > (self.fixation_radius + self.fixation_radius_buffer)

    def _test_fixation_complete(self, ts):
        '''
        Test whether the fixation is complete. In sequence trials, there are only 1 fixation period for the center target. After subjects fixate
        the peripheral eye target, fixation state is skipped and the next delay starts.
        In simultaneous trials, there are 2 fixation periods for the center target and the peripheral target
        '''
        if self.target_index == 0:
            return ts > self.fixation_time
        elif self.target_index == 1 and self.is_simultaneous:
            return ts > self.fixation_time
        else:
            return True
    
    def _test_hold_complete(self, ts):
        return ts > self.hold_time
    
    def _test_delay_complete(self, ts):
        '''
        Test whether the delay period is over. In sequence trials, there are 2 delay period for each eye and hand.
        In simultaneous trials, there is only 1 delay period.
        '''
        if self.target_index == 0 and self.is_simultaneous:
            return ts > self.delay_time_eye_hand
        elif self.target_index == 0 and self.is_sequence:
            return ts > self.delay_time_eye
        elif self.target_index == 1 and self.is_sequence:
            return ts > self.delay_time_hand
        else:
            return True
    
    def _test_trial_complete(self, ts):
        return self.eye_target_index == 1 and self.hand_target_index == 1
    
    def _test_targ_simultaneous(self,ts):
        return self.eye_target_index == 0 and self.hand_target_index == 0 and self.is_simultaneous
    
    def _test_targ_first_sequence(self, ts):
        return self.eye_target_index == 0 and self.hand_target_index == 0 and self.is_sequence
    
    def _test_targ_second_sequence(self, ts):
        return self.eye_target_index == 1 and self.hand_target_index == 0 and self.is_sequence
    
    def _test_coordination_penalty_end(self, ts):
        return ts > self.coordination_penalty_time

    def _test_tooslow_penalty_end(self, ts):
        return ts > self.tooslow_penalty_time
    
    def _start_wait(self):
        super()._start_wait()

        # Initialize target index and target positons for eye and hand
        self.eye_target_index = -1
        self.hand_target_index = -1
        self.eye_targs = np.copy(self.targs)
        self.hand_targs = np.copy(self.targs)
        self.eye_gen_indices = np.copy(self.gen_indices)
        self.hand_gen_indices = np.copy(self.gen_indices)

        if self.calc_trial_num() == 0:

            # Instantiate the targets here so they don't show up in any states that might come before "wait"
            for target in self.targets_eye:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

            for target in self.targets_hand:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

        if self.tries == 0: # Update delay_time only in the first attempt

            # Decide sequence or simultaneous trials  
            self.trial_count_blocks = self.calc_state_occurrences('reward') % self.trials_all_blocks

            if self.trial_count_blocks < self.trials_block_simultaneous:
                self._set_trial_type(is_simultaneous=True)

            elif self.trial_count_blocks - self.trials_block_simultaneous < self.trials_block_sequence:
                self._set_trial_type(is_simultaneous=False)

            self.task_data['is_sequence'] = self.is_sequence

        if self.is_sequence:
            for target in self.targets_hand:
                target.sphere.color = target_colors[self.sequence_target_color]

        # For eye calibration during the task
        if self.trials_online_eye_calib:
            self.eye_pos_tmp0 = []
            self.eye_pos_tmp1 = []

            if self.tries == 0:
                if self.calc_state_occurrences('reward') > self.trials_online_eye_calib:
                    if len(self.eye_pos) == 4:
                        target_pos_tile = np.tile(np.array(self.targ_pos_online_eye_calib), (1,2))
                    else:
                        target_pos_tile = np.array(self.targ_pos_online_eye_calib)

                    self.eye_center = np.mean(self.m_eye_pos_center_online_eye_calib, axis=0)
                    slopes, intercepts, _ = aopy.analysis.fit_linear_regression(np.array(self.m_eye_pos_online_eye_calib)-self.eye_center, target_pos_tile)
                    self.eye_coeff = np.vstack((slopes, intercepts)).T
                    self.eye_coeff_stack.append(self.eye_coeff)

    def _set_trial_type(self, is_simultaneous):
        if is_simultaneous:
            self.is_simultaneous = True
            self.is_sequence = False

            s, e = self.rand_delay_eye_hand
            self.delay_time_eye_hand = random.random()*(e-s) + s

            self.chain_length = 2
            self.reaction_time_thr = self.hand_RTs_thr_simul

            self.reward_time = self.reward_time_simultaneous
            if self.reward_count_simultaneous % self.trials_jackpot_simultaneous == self.trials_jackpot_simultaneous-1:
                self.reward_time = self.reward_time_jackpot_simultaneous

        else:
            self.is_simultaneous = False
            self.is_sequence = True
            
            s, e = self.rand_delay_eye
            self.delay_time_eye = random.random()*(e-s) + s
            s, e = self.rand_delay_hand
            self.delay_time_hand = random.random()*(e-s) + s

            self.chain_length = 3
            self.reaction_time_thr = self.hand_RTs_thr_seq

            self.reward_time = self.reward_time_sequence
            if self.reward_count_sequence % self.trials_jackpot_sequence == self.trials_jackpot_sequence-1:
                self.reward_time = self.reward_time_jackpot_sequence

    def _start_target(self):
        
        self.target_index += 1
        self.eye_target_index += 1
        self.hand_target_index += 1

        # Show eye hand target
        target_hand = self.targets_hand[self.hand_target_index]
        target_hand.move_to_position(self.hand_targs[self.hand_target_index])
        target_eye = self.targets_eye[self.eye_target_index]
        target_eye.move_to_position(self.eye_targs[self.eye_target_index] - self.offset_cube)
        target_hand.show()
        target_eye.show()
        self.sync_event('EYE_TARGET_ON', self.eye_gen_indices[self.eye_target_index])

    def _while_target(self):
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])
        if eye_d <= (self.target_radius + self.fixation_radius_buffer):
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.fixation_target_color] # chnage color in fixating center
        else:
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.eye_target_color]
    
    def _start_target_eye(self):
        self.target_index += 1
        self.eye_target_index += 1

    def _while_target_eye(self):
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])
        if eye_d <= (self.target_radius + self.fixation_radius_buffer):
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.fixation_target_color] # chnage color in fixating center
        else:
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.eye_target_color]

    def _start_target_hand(self):
        self.target_index += 1
        self.hand_target_index += 1

        # for reaction time detection
        self.reaction_time_hand = 0

    def _start_target_eye_hand(self):
        self.target_index += 1
        self.eye_target_index += 1
        self.hand_target_index += 1

        # for reaction time detection
        self.reaction_time_hand = 0
        self.reaction_time_eye = 0

    def _while_target_eye_hand(self):
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])
        if eye_d <= (self.target_radius + self.fixation_radius_buffer):
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.fixation_target_color] # chnage color in fixating center
        else:
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.eye_target_color]

    def _start_fixation(self):
        self.sync_event('FIXATION', self.eye_gen_indices[self.eye_target_index])
        self.targets_eye[self.eye_target_index].cube.color = target_colors[self.fixation_target_color]

        if self.trials_online_eye_calib:
            self.start_fixation_time = self.get_time()

    def _while_fixation(self):
        # Only store eye pos between offset ~ offset + duration in the fixation state
        if self.trials_online_eye_calib:
            elapsed_time = self.get_time() - self.start_fixation_time
            if elapsed_time > self.offset_time_online_eye_calib and elapsed_time < self.offset_time_online_eye_calib + self.duration_online_eye_calib:
                if self.eye_target_index == 0:
                    self.eye_pos_tmp0.append(self.eye_pos[:4]) # store eye data for the center target
                elif self.eye_target_index == 1 and self.is_simultaneous:
                    self.eye_pos_tmp1.append(self.eye_pos[:4]) # store eye data for the peripheral target from eye trials

    def _start_hold(self):
        if self.target_index != 1: # when the state comes from target_eye, skip start_hold
            self.sync_event('CURSOR_ENTER_TARGET', self.hand_gen_indices[self.hand_target_index])

        if self.trials_online_eye_calib:
            self.start_hold_time = self.get_time()

    def _while_hold(self):
        # Only store eye pos between offset ~ offset + duration in the hold state of eye-hand trials
        if self.trials_online_eye_calib:
            elapsed_time = self.get_time() - self.start_hold_time
            if elapsed_time > self.offset_time_online_eye_calib and elapsed_time < self.offset_time_online_eye_calib + self.duration_online_eye_calib:
                if self.hand_target_index == 1 and self.is_sequence:
                    self.eye_pos_tmp1.append(self.eye_pos[:4]) # store eye data for the peripheral target from sequential trials

    def _start_delay(self):
        if self.target_index == 0 and self.is_simultaneous: # This is for both eye and hand targets
            next_idx = (self.eye_target_index + 1)
            self.targets_eye[next_idx].move_to_position(self.eye_targs[next_idx] - self.offset_cube)
            self.targets_hand[next_idx].move_to_position(self.hand_targs[next_idx])
            self.targets_eye[next_idx].show()
            self.targets_hand[next_idx].show()
            self.sync_event('EYE_TARGET_ON', self.eye_gen_indices[next_idx])

        elif self.target_index == 0 and self.is_sequence: # This is for eye target in the first delay
            next_eye_idx = (self.eye_target_index + 1)
            self.targets_eye[next_eye_idx].move_to_position(self.eye_targs[next_eye_idx] - self.offset_cube)
            self.targets_eye[next_eye_idx].show()
            
            next_hand_idx = (self.hand_target_index + 1)
            self.targets_hand[next_hand_idx].move_to_position(self.hand_targs[next_hand_idx]) # Target position is the same, but change color?
            self.targets_hand[next_hand_idx].show()

            self.sync_event('EYE_TARGET_ON', self.eye_gen_indices[next_eye_idx])

    def _start_targ_transition(self):
        if self.target_index == 0 and self.is_simultaneous: # This is a go cue for both eye and hand
            self.targets_eye[self.eye_target_index].hide()
            self.targets_hand[self.hand_target_index].hide()
            self.sync_event('EYE_TARGET_OFF', self.eye_gen_indices[self.eye_target_index])

        elif self.target_index == 0 and self.is_sequence: # This is a go cue for eye
            self.targets_eye[self.eye_target_index].hide()
            self.sync_event('EYE_TARGET_OFF', self.eye_gen_indices[self.eye_target_index])
        
        elif self.target_index == 1 and self.is_sequence: # This is a go cue for hand
            self.targets_hand[self.hand_target_index].hide()
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.sequence_gocue_color]
            self.sync_event('TARGET_OFF', self.hand_gen_indices[self.hand_target_index])   

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()
            
    def _start_hold_penalty(self):
        super()._start_hold_penalty()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()

    def _start_delay_penalty(self):
        super()._start_delay_penalty()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()

    def _start_fixation_penalty(self):
        super()._start_fixation_penalty()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()

    def _start_coordination_penalty(self):
        self._increment_tries()
        self.sync_event('OTHER_PENALTY')
        self.penalty_index = 1

        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()

    def _end_coordination_penalty(self):
        self.sync_event('TRIAL_END')

    def _start_tooslow_penalty(self):
        self._increment_tries()
        self.sync_event('OTHER_PENALTY')
        self.penalty_index = 1

        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()

    def _end_tooslow_penalty(self):
        self.sync_event('TRIAL_END')

    def _start_reward(self):
        super()._start_reward()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.cue_trial_end_success()
            target_hand.cue_trial_end_success()

        if self.is_sequence:
            self.reward_count_sequence += 1
        else:
            self.reward_count_simultaneous += 1

    def _end_reward(self):
        super()._end_reward()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()     

        if self.trials_online_eye_calib:
            self.m_eye_pos_center_online_eye_calib.append(np.nanmean(self.eye_pos_tmp0, axis=0)) # eye data for the center target
            self.m_eye_pos_online_eye_calib.append(np.nanmean(self.eye_pos_tmp1, axis=0)) # eye data for the peripheral target
            self.targ_pos_online_eye_calib.append(np.array(self.targs)[1,[0,2]]) # periphetal target position

    def _start_pause(self):
        super()._start_pause()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()

    def cleanup_hdf(self):
        super().cleanup_hdf()
        if self.trials_online_eye_calib:
            if hasattr(self, "h5file"):
                h5file = tables.open_file(self.h5file.name, mode='a')
                h5file.root.task.attrs['eye_coeff_stack'] = np.array(self.eye_coeff_stack)
                h5file.close()

class ScreenTargetCapture_Saccade(ScreenTargetCapture):
    '''
    Center-out saccade task. The controller for the cursor position is eye position.
    Hand cursor is invisible.
    '''

    fixation_radius_buffer = traits.Float(.5, desc="additional radius for eye target")
    target_color = traits.OptionsList("eye_color", *target_colors, desc="Color of the target", bmi3d_input_options=list(target_colors.keys()))
    fixation_target_color = traits.OptionsList("fixation_color", *target_colors, desc="Color of the eye target under fixation state", bmi3d_input_options=list(target_colors.keys()))
    automatic_reward = traits.Bool(False, desc="Whether to deliver automatic reward")

    init_eye_target_alpha = traits.Float(1., desc="Transparency of initial eye targets")
    goal_eye_target_alpha = traits.Float(0.3, desc="Transparency of initial hand targets")

    sequence_generators = [
        'out_2D', 'out_2D_select', 'centerout_2D', 'centeroutback_2D', 'centerout_2D_select', 'rand_target_chain_2D', 'rand_same_target_chain_2D', 
        'rand_target_chain_3D', 'corners_2D', 'centerout_tabletop', 'out_2D_square', 'centerout_2D_square', 'row_target','sac_hand_2d'
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:

            # Control transparency of targets
            new_color1 = list(target_colors[self.target_color])
            new_color1[3] = self.init_eye_target_alpha
            new_color2 = list(target_colors[self.target_color])
            new_color2[3] = self.goal_eye_target_alpha

            # 2 targets for delay
            target1 = VirtualRectangularTarget(target_width=self.target_radius, target_height=self.target_radius/2, target_color=new_color1)
            target2 = VirtualRectangularTarget(target_width=self.target_radius, target_height=self.target_radius/2, target_color=new_color2)

            self.targets = [target1, target2]

            self.offset_cube = np.array([0,0,self.target_radius/2]) # To center the cube target

    def _start_wait(self):
        super()._start_wait()
        # Make cursor invisible
        self.plant.cursor.detach()
        self.chain_length = len(self.targs)
        
    def _test_enter_target(self, ts):
        '''
        Check whether eye positions from a target are within the fixation distance
        '''
        # Distance of an eye position from a target position
        if not self.automatic_reward:
            eye_pos = self.calibrated_eye_pos
            target_pos = np.delete(self.targs[self.target_index],1)
            d_eye = np.linalg.norm(eye_pos - target_pos)
            return (d_eye <= self.target_radius + self.fixation_radius_buffer)
        else:
            return True

    def _test_leave_target(self, ts):
        '''
        Check whether eye positions from a target are outside the fixation distance
        '''
        # Distance of an eye position from a target position
        if not self.automatic_reward:
            eye_pos = self.calibrated_eye_pos
            target_pos = np.delete(self.targs[self.target_index],1)
            d_eye = np.linalg.norm(eye_pos - target_pos)
            return (d_eye > self.target_radius + self.fixation_radius_buffer)
        else:
            return False

    def _start_target(self):
        self.target_index += 1

        # Show target if it is hidden (this is the first target, or previous state was a penalty)
        target = self.targets[self.target_index % 2]
        if self.target_index == 0:
            target.move_to_position(self.targs[self.target_index] - self.offset_cube)
            target.show()
            self.sync_event('TARGET_ON', self.gen_indices[self.target_index])
        self.target_location = self.targs[self.target_index] # save for BMILoop

    def _start_hold(self):
        super()._start_hold()
        self.targets[self.target_index].cube.color = target_colors[self.fixation_target_color] # change target color in fixating the target

    def _start_delay(self):
        # Make next target visible unless this is the final target in the trial
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length:
            target = self.targets[next_idx % 2]
            target.move_to_position(self.targs[next_idx] - self.offset_cube)
            target.show()
            self.sync_event('TARGET_ON', self.gen_indices[next_idx])
        else:
            # This delay state should only last 1 cycle, don't sync anything
            pass

    # Generator functions
    @staticmethod
    def row_target(nblocks=20, ntargets=3, dx=5.,offset1=(0,0,-2),offset2=(0,0,6.),offset3=(0,0,-7.5),origin=(0,0,0)):
        '''
        Generates a sequence of 3D for 2 eye targets and 1 hand target at a given distance from the origin

        Parameters
        ----------
        nblocks : int
            The number of ntarget pairs in the sequence.
        ntargets : int
            The number of equally spaced targets
        distance : float
            The distance in cm between targets
        offset1 : 3-tuple
            y location of the first eye target
        offset2 : 3-tuple
            y location of the second eye target
        offset3 : 3-tuple
            y location of the hand target
        origin : 3-tuple
            Location of the central targets

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [1 x 3] target coordinates

        '''
        rng = np.random.default_rng()
        for _ in range(nblocks):
            order = np.arange(ntargets**3)
            rng.shuffle(order)
            x_pos_candidate = [-dx,0,dx]
            for t in range(ntargets**3):
                idx = np.base_repr(order[t],3).zfill(3) # convert a decimal number to ternary

                # Target index for hand target, initial eye target, final eye target
                idx1 = int(idx[0])
                idx2 = int(idx[1])
                idx3 = int(idx[2])

                # Get positions for each target
                x_pos1 = x_pos_candidate[idx1]
                x_pos2 = x_pos_candidate[idx2]
                x_pos3 = x_pos_candidate[idx3]
                pos1 = np.array([x_pos1,0,0]).T
                pos2 = np.array([x_pos2,0,0]).T
                pos3 = np.array([x_pos3,0,0]).T

                yield [idx1],[idx2],[idx3],[pos1+offset1+origin],[pos2+offset2+origin],[pos3+offset3+origin]

    @staticmethod
    def sac_hand_2d(nblocks=20, ntargets=3, dx=10,offset1=(0,0,-2),offset2=(0,0,6.),offset3=(0,0,-7.5),origin=(0,0,0)):
        '''
        Pairs of hand targets and eye targets

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [3 x 3] target coordinates
        '''

        gen = HandConstrainedEyeCapture.row_target(nblocks=nblocks,ntargets=ntargets,dx=dx,offset1=offset1,offset2=offset2,offset3=offset3,origin=origin)
        for _ in range(nblocks*(ntargets**3)):
            idx1,idx2,idx3,pos1,pos2,pos3 = next(gen)

            targs = np.zeros([2, 3])
            targs[0,:] = pos1[0]
            targs[1,:] = pos2[0]
            #targs[2,:] = pos3[0]

            indices = np.zeros([2,1])
            indices[0] = idx1[0]
            indices[1] = idx2[0] + ntargets
            #indices[2] = idx3[0]

            yield indices, targs

class ScreenTargetCapture_Saccade_withBackgroundTargets(ScreenTargetCapture_Saccade):
    background_target_radius = traits.Float(2, desc="Radius of targets in cm")
    background_target_color = traits.OptionsList("yellow", *target_colors, desc="Color of the target", bmi3d_input_options=list(target_colors.keys()))
    show_pheriheral_square_target = traits.Bool(True, desc="Whether to show peripheral square target")
    def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            instantiate_targets = kwargs.pop('instantiate_targets', True)
            if instantiate_targets:
                print("Instantiating background targets")
                background_target = VirtualCircularTarget(target_radius=self.background_target_radius, target_color=target_colors[self.background_target_color])
                self.targets.append(background_target)

    def _start_delay(self):
            # Make next target visible unless this is the final target in the trial
            next_idx = (self.target_index + 1)
            if next_idx < self.chain_length:
                target = self.targets[next_idx % 2]
                bck_target = self.targets[-1]
                sphere_position = self.targs[next_idx] - [0 ,-10,0]
                cube_position = self.targs[next_idx] - self.offset_cube
                target.move_to_position(cube_position)
                bck_target.move_to_position(sphere_position)

                if self.show_pheriheral_square_target:
                    target.show()
                bck_target.show()

                self.sync_event('TARGET_ON', self.gen_indices[next_idx])
            else:
                # This delay state should only last 1 cycle, don't sync anything
                pass


    
    def _start_reward(self):
        #self.targets[self.target_index % 2].cue_trial_end_success()
        self.sync_event('REWARD')