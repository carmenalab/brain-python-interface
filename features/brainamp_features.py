'''
Features for acquiring data from the BrainAmp EMG recording system (model name?)
'''

import time

from riglib.experiment import traits
from riglib.ismore import settings
import numpy as np


class BrainAmpData(traits.HasTraits):
    '''Stream BrainAmp EMG/EEG/EOG data.'''

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up a MultiChanDataSource to stream data from the BrainAmp system
        '''
        from riglib import source
        from riglib.brainamp import rda

        self.brainamp_channels = settings.BRAINAMP_CHANNELS
        self.brainamp_source = source.MultiChanDataSource(rda.EMGData, 
            name='brainamp', channels=self.brainamp_channels, send_data_to_sink_manager=True)

        from riglib import sink
        sink.sinks.register(self.brainamp_source)

        super(BrainAmpData, self).init()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the EMG data source
        '''
        
        self.brainamp_source.start()
        self.ts_start_brainamp = time.time()
        try:
            super(BrainAmpData, self).run()
        finally:
            self.brainamp_source.stop()

    
            
    def _cycle(self):
        if settings.VERIFY_BRAINAMP_DATA_ARRIVAL:
            self.verify_brainamp_data_arrival(settings.VERIFY_BRAINAMP_DATA_ARRIVAL_TIME)

        super(BrainAmpData, self)._cycle()

    def verify_brainamp_data_arrival(self, n_secs):
        time_since_brainamp_started = time.time() - self.ts_start_brainamp
        last_ts_arrival = self.last_brainamp_data_ts_arrival()
     
        if time_since_brainamp_started > n_secs:
            if last_ts_arrival == 0:
                print 'No BrainAmp data has arrived at all'
            else:
                t_elapsed = time.time() - last_ts_arrival
                if t_elapsed > n_secs:
                    print 'No BrainAmp data has arrived in the last %.1f s' % t_elapsed

    def last_brainamp_data_ts_arrival(self):
        return np.max(self.brainamp_source.get(n_pts=1, channels=self.brainamp_source.channels)['ts_arrival'])
