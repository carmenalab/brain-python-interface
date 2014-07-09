import numpy as np

import cerelink
import udp_feedback_client


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



class FeedbackData(object):
    '''Abstract parent class, not meant to be instantiated.'''

    client_cls = None

    def __init__(self):
        self.client = self.client_cls()

    def start(self):
        self.client.start()
        self.data = self.client.get_feedback_data()

    def stop(self):
        self.client.stop()

    def get(self):
        d = self.data.next()

        return np.array([(tuple(d.data), tuple(d.ts), d.arrival_ts)], dtype=self.dtype)

    @staticmethod
    def _get_dtype(state_names):
        sub_dtype_data = np.dtype([(state_name, np.float64) for state_name in state_names])
        sub_dtype_ts   = np.dtype([(state_name, np.int64) for state_name in state_names])
        return np.dtype([('data', sub_dtype_data), ('ts', sub_dtype_ts), ('arrival_ts', np.float64)])


# for use with a DataSource
class ArmAssistData(FeedbackData):
    update_freq = 25.  # TODO check
    client_cls = udp_feedback_client.ArmAssistClient

    state_names = ['aa_px', 'aa_py', 'aa_ppsi']
    dtype = FeedbackData._get_dtype(state_names)


# for use with a DataSource
class ReHandData(FeedbackData):
    update_freq = 25.  # TODO check
    client_cls = udp_feedback_client.ReHandClient

    state_names = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono', 'rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']
    dtype = FeedbackData._get_dtype(state_names)

