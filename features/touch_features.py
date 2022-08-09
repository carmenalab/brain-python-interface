
'''
Features for a touch sensor on the neurosync arduino
'''
from riglib.experiment import traits
from riglib import touch_data

########################################################################################################
# Touch sensor datasources
########################################################################################################

class TouchDataFeature(traits.HasTraits):
    '''
    Enable reading of data from touch sensor
    '''

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the DataSource for interacting with the 
        kinarm system and registers the source with the SinkRegister so that the data gets saved 
        to file as it is collected.
        '''
        from riglib import source
        System  = touch_data.TouchData
        self.touch_data = source.DataSource(System)
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.touch_data)
        super(TouchDataFeature, self).init()
    
    @property
    def source_class(self):
        '''
        Specify the source class as a function
        '''
        from riglib import touch_data
        return touch_data.TouchData()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the motiontracker source prior to starting the experiment's 
        main thread/process, and handle any errors by stopping the source
        '''
        self.touch_data.start()
        try:
            super(TouchDataFeature, self).run()
        finally:
            self.touch_data.stop()
    
    def join(self):
        '''
        See riglib.experiment.Experiment.join(). Re-join the 'motiondata' source process before cleaning up the experiment thread
        '''
        #self.touch_data.join()
        super(TouchDataFeature, self).join()
    
    def _start_None(self):
        '''
        Code to run before the 'None' state starts (i.e., the task stops)
        '''
        self.touch_data.stop()
        super(TouchDataFeature, self)._start_None()
