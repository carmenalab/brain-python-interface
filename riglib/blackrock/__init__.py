import numpy as np

import cerelink


# for use with a DataSource
class Spikes(object):
    update_freq = 30000.  # TODO -- double check
    dtype = np.dtype([("ts", np.float), ("chan", np.int32), ("unit", np.int32), ("arrival_ts", np.float64)])

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
        
        return np.array([(d.ts / self.update_freq, d.chan, d.unit, d.arrival_ts)], dtype=self.dtype)


# for use with a MultiChanDataSource
class LFP(object):
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

        # TODO - need to convert samples to mV first?
        return (d.chan, d.samples)
