'''
Features for use in simulation tasks
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

class FakeHDF(object):
    def __init__(self):
        self.msgs = []

    def sendMsg(self, msg):
        self.msgs.append(msg)


class SimHDF(object):
    '''
    An interface-compatbile HDF for simulations which do not require saving an
    HDF file
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor for SimHDF feature

        Parameters
        ----------
        args, kwargs: None necessary

        Returns
        -------
        SimHDF instance
        '''
        from collections import defaultdict
        self.data = defaultdict(list)
        self.task_data_hist = []
        self.msgs = []        
        self.hdf = FakeHDF()

        super(SimHDF, self).__init__(*args, **kwargs)

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' creates a fake task data variable so that 
        code expecting SaveHDF runs smoothly.
        '''
        super(SimHDF, self).init()
        self.dtype = np.dtype(self.dtype)
        self.task_data = np.zeros((1,), dtype=self.dtype)

    def sendMsg(self, msg):
        '''
        Simulate the "message" table of the HDF file associated with each source

        Parameters
        ----------
        msg: string
            Message to store

        Returns
        -------
        None
        '''
        self.msgs.append((msg, -1))

    def _cycle(self):
        super(SimHDF, self)._cycle()
        self.task_data_hist.append(self.task_data.copy())


class SimTime(object):
    '''
    An accelerator so that simulations can run faster than real time (the task doesn't try to 'sleep' between loop iterations)
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(SimTime, self).__init__(*args, **kwargs)
        self.start_time = 0.

    def get_time(self):
        ''' Docstring '''
        try:
            return self.cycle_count * self.update_rate
        except:
            # loop_counter has not been initialized yet, return 0
            return 0

    @property 
    def update_rate(self):
        '''
        Attribute for update rate of task. Using @property in case any future modifications
        decide to change fps on initialization
        '''
        return 1./60