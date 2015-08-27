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

###### CONSTANTS
sec_per_min = 60

class AdaptiveGenerator(object):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(AdaptiveGenerator, self).__init__(*args, **kwargs)
        assert hasattr(self.gen, "correct"), "Must use adaptive generator!"

    def _start_reward(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.gen.correct()
        super(AdaptiveGenerator, self)._start_reward()
    
    def _start_incorrect(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.gen.incorrect()
        super(AdaptiveGenerator, self)._start_incorrect()


class IgnoreCorrectness(object):
    '''Allows any response to be correct, not just the one defined. Overrides for trialtypes'''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(IgnoreCorrectness, self).__init__(*args, **kwargs)
        if hasattr(self, "trial_types"):
            for ttype in self.trial_types:
                del self.status[ttype]["%s_correct"%ttype]
                del self.status[ttype]["%s_incorrect"%ttype]
                self.status[ttype]["correct"] = "reward"
                self.status[ttype]["incorrect"] = "penalty"

    def _test_correct(self, ts):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return self.event is not None

    def _test_incorrect(self, ts):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return False


class Autostart(traits.HasTraits):
    '''Automatically begins the trial from the wait state, with a random interval drawn from `rand_start`'''
    rand_start = traits.Tuple((0., 0.), desc="Start interval")
    wait_time = 0

    def _start_wait(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        s, e = self.rand_start
        self.wait_time = random.random()*(e-s) + s
        super(Autostart, self)._start_wait()
        
    def _test_start_trial(self, ts):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return ts > self.wait_time and not self.pause
    
    def _test_premature(self, ts):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return self.event is not None

