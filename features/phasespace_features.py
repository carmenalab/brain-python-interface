'''
Features for the phasephase motiontracker
'''

import time
import tempfile
import random
import traceback
import numpy as np
import fnmatch
import os

from riglib import calibrations

import os
import subprocess

import time
from riglib.experiment import traits

########################################################################################################
# Phasespace datasources
########################################################################################################
class MotionData(traits.HasTraits):
    '''
    Enable reading of raw motiontracker data from Phasespace system
    '''
    marker_count = traits.Int(8, desc="Number of markers to return")

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the DataSource for interacting with the 
        motion tracker system and registers the source with the SinkRegister so that the data gets saved to file as it is collected.
        '''
        from riglib import source
        src, mkw = self.motion_source
        self.motiondata = source.DataSource(src, **mkw)
        from riglib import sink
        self.sinks = sink.sinks
        self.sinks.register(self.motiondata)
        super(MotionData, self).init()
    
    @property
    def motion_source(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import motiontracker
        return motiontracker.make(self.marker_count), dict()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the motiontracker source prior to starting the experiment's 
        main thread/process, and handle any errors by stopping the source
        '''
        self.motiondata.start()
        try:
            super(MotionData, self).run()
        finally:
            self.motiondata.stop()
    
    def join(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.motiondata.join()
        super(MotionData, self).join()
    
    def _start_None(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.motiondata.stop()
        super(MotionData, self)._start_None()

class MotionSimulate(MotionData):
    '''
    Simulate presence of raw motiontracking system using a randomized spatial function
    '''
    @property
    def motion_source(self):
        from riglib import motiontracker
        cls = motiontracker.make(self.marker_count, cls=motiontracker.Simulate)
        return cls, dict(radius=(100,100,50), offset=(-150,0,0))

class MotionAutoAlign(MotionData):
    '''Creates an auto-aligning motion tracker, for use with the 6-point alignment system'''
    autoalign = traits.Instance(calibrations.AutoAlign)
    
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' adds a filter onto the motiondata source. See MotionData for further details.
        '''
        super(MotionAutoAlign, self).init()
        self.motiondata.filter = self.autoalign

    @property
    def motion_source(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import motiontracker
        cls = motiontracker.make(self.marker_count, cls=motiontracker.AligningSystem)
        return cls, dict()
