'''Docstring.'''

import numpy as np

from config import config
if config.recording_sys['make'] == 'blackrock':
    import cerelink
    


class Spikes(object):
    '''For use with a DataSource in order to acquire streaming spike data from 
    the Blackrock Neural Signal Processor (NSP).
    '''

    update_freq = 30000.
    dtype = np.dtype([("ts", np.float), 
                      ("chan", np.int32), 
                      ("unit", np.int32),
                      ("arrival_ts", np.float64)])

    def __init__(self, channels):
        self.conn = cerelink.Connection()
        self.conn.connect()
        self.conn.select_channels(channels)

    def start(self):
        self.conn.start_data()
        self.data = self.conn.get_event_data()

    def stop(self):
        self.conn.stop_data()

    def get(self):
        d = self.data.next()
        return np.array([(d.ts / self.update_freq, 
                          d.chan, 
                          d.unit, 
                          d.arrival_ts)],
                        dtype=self.dtype)


class LFP(object):
    '''For use with a MultiChanDataSource in order to acquire streaming LFP 
    data from the Blackrock Neural Signal Processor (NSP).
    '''
    
    update_freq = 2000  # TODO -- change back to 1000 
    dtype = np.dtype('float')

    def __init__(self, channels):
        self.conn = cerelink.Connection()
        self.conn.connect()
        self.conn.select_channels(channels)

    def start(self):
        self.conn.start_data()
        self.data = self.conn.get_continuous_data()

    def stop(self):
        self.conn.stop_data()

    def get(self):
        d = self.data.next()
        return (d.chan, d.samples)  # TODO -- document the units (mV?)
