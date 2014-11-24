'''
Features for acquiring data from the BrainAmp EMG recording system (model name?)
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


class BrainAmpData(traits.HasTraits):
    '''Stream BrainAmp neural data.'''

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up a MultiChanDataSource to stream data from the BrainAmp system
        '''
        from riglib import brainamp, source

        self.emgdata = source.MultiChanDataSource(brainamp.EMG, channels=channels)

        try:
            super(BrainAmpData, self).init()
        except:
            print "BrainAmpData: running without a task"

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the EMG data source
        '''        
        self.emgdata.start()
