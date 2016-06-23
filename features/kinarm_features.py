'''
Features for the kinarm which is sending packets blindly to the BMI machine
'''

import time
import random
import traceback
import numpy as np

import os
import subprocess

import time
from riglib.experiment import traits

########################################################################################################
# Phasespace datasources
########################################################################################################
class KinarmData(traits.HasTraits):
    '''
    Enable reading of data from Kinarm (3 x 50 matrix)
    '''

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the DataSource for interacting with the 
        kinarm system and registers the source with the SinkRegister so that the data gets saved 
        to file as it is collected.
        '''
        from riglib import source
        src, mkw = self.source_class
        self.kinarmdata = source.DataSource(src, **mkw)
        from riglib import sink
        self.sinks = sink.sinks
        self.sinks.register(self.kinarmdata)
        super(MotionData, self).init()
    
    @property
    def source_class(self):
        '''
        Specify the source class as a function
        '''
        from riglib import kinarmdata
        return kinarmdata.KinarmData()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the motiontracker source prior to starting the experiment's 
        main thread/process, and handle any errors by stopping the source
        '''
        self.kinarmdata.start()
        try:
            super(KinarmData, self).run()
        finally:
            self.kinarmdata.stop()
    
    def join(self):
        '''
        See riglib.experiment.Experiment.join(). Re-join the 'motiondata' source process before cleaning up the experiment thread
        '''
        self.kinarmdata.join()
        super(KinarmData, self).join()
    
    def _start_None(self):
        '''
        Code to run before the 'None' state starts (i.e., the task stops)
        '''
        self.kinarmdata.stop()
        super(KinarmData, self)._start_None()


class MotionSimulate(MotionData):
    '''
    Simulate presence of raw motiontracking system using a randomized spatial function
    '''
    @property
    def source_class(self):
        '''
        Specify the source class as a function in case future descendant classes want to use a different type of source
        '''        
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
    def source_class(self):
        '''
        Specify the source class as a function in case future descendant classes want to use a different type of source
        '''
        from riglib import motiontracker
        cls = motiontracker.make(self.marker_count, cls=motiontracker.AligningSystem)
        return cls, dict()
