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
    with a random interval drawn from `rand_start`
    '''
    rand_start = traits.Tuple((0., 0.), desc="Start interval")

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
