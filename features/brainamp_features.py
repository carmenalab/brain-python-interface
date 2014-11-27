'''
Features for acquiring data from the BrainAmp EMG recording system (model name?)
'''

from riglib.experiment import traits


class BrainAmpData(traits.HasTraits):
    '''Stream BrainAmp EMG/EEG/EOG data.'''

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up a MultiChanDataSource to stream data from the BrainAmp system
        '''
        from riglib import source
        from riglib.brainamp import rda
        self.emgdata = source.MultiChanDataSource(rda.EMGData, name='emg', channels=channels, send_data_to_sink_manager=True)

        from riglib import sink
        sink.sinks.register(self.emgdata)

        super(BrainAmpData, self).init()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the EMG data source
        '''
        self.emgdata.start()
