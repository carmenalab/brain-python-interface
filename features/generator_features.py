'''
Features which have task-like functionality w.r.t. task...
'''

import time
import tempfile
import random
import traceback
import numpy as np
import fnmatch
import os
import subprocess
from riglib.experiment import traits


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

    hold_time = traits.List([.2,], desc="Length of hold required at targets before next target appears. \
        Can be a single number or a list of numbers to apply to each target in the sequence (center, out, etc.)")

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
        if len(self.hold_time) == 1:
            hold_time = self.hold_time[0]
        else:
            hold_time = self.hold_time[self.target_index]
        return time_in_state > hold_time

class RandomDelay(traits.HasTraits):
    '''
    Replaces 'delay_time' with 'rand_delay', an interval on which the delay period is selected uniformly.
    '''
    
    rand_delay = traits.Tuple((0., 0.), desc="Delay interval")
    exclude_parent_traits = ['delay_time']

    def _start_wait(self):
        '''
        At the start of the 'wait' state, draw a sample from the rand_delay interval for this trial.
        '''
        s, e = self.rand_delay
        self.delay_time = random.random()*(e-s) + s
        super()._start_wait()

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
            self._old_target_color = np.copy(target.sphere.color)
            new_target_color = list(target.sphere.color)
            new_target_color[3] = self.delay_target_alpha
            target.sphere.color = tuple(new_target_color)

    def _start_target(self):
        super()._start_target()

        # Reset the transparency of the current target
        if self.target_index > 0:
            target = self.targets[self.target_index % 2]
            target.sphere.color = self._old_target_color

