'''
Extensions of the generic riglib.source.DataSourceSystem for getting Spikes/LFP data from the Blackrock NeuroPort system over the rig's internal network (UDP)
'''
import numpy as np
from ..source import DataSourceSystem

class Spikes(DataSourceSystem):
    '''
    For use with a DataSource in order to acquire streaming spike data from 
    the Blackrock Neural Signal Processor (NSP).
    '''

    update_freq = 30000.
    dtype = np.dtype([("ts", np.float), 
                      ("chan", np.int32), 
                      ("unit", np.int32),
                      ("arrival_ts", np.float64)])

    def __init__(self, channels):
        from . import cerelink
        self.conn = cerelink.Connection()
        self.conn.connect()
        self.conn.select_channels(channels)

    def start(self):
        self.conn.start_data()
        self.data = self.conn.get_event_data()

    def stop(self):
        self.conn.stop_data()

    def get(self):
        d = next(self.data)
        return np.array([(d.ts / self.update_freq, 
                          d.chan, 
                          d.unit, 
                          d.arrival_ts)],
                        dtype=self.dtype)


class LFP(DataSourceSystem):
    '''
    For use with a MultiChanDataSource in order to acquire streaming LFP 
    data from the Blackrock Neural Signal Processor (NSP).
    '''
    
    update_freq = 1000
    dtype = np.dtype('float')

    def __init__(self, channels):
        from . import cerelink
        self.conn = cerelink.Connection()
        self.conn.connect()
        self.conn.select_channels(channels)

    def start(self):
        self.conn.start_data()
        self.data = self.conn.get_continuous_data()

    def stop(self):
        self.conn.stop_data()

    def get(self):
        d = next(self.data)
        return (d.chan, d.samples)
