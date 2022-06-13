'''
Features for the kinarm which is sending packets blindly to the BMI machine
'''
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
        from riglib import kinarmdata
        System = Motion = kinarmdata.Kinarmdata
        self.kinarmdata = source.DataSource(System)
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.kinarmdata)
        super(KinarmData, self).init()
    
    @property
    def source_class(self):
        '''
        Specify the source class as a function
        '''
        from riglib import kinarmdata
        return kinarmdata.Kinarmdata()

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
