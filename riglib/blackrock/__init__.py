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
    update_freq = 1000.  # TODO -- double check

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


# old
# use this class if using only one data source for armassist+rehand
# for use with a MultiChanDataSource
class FeedbackData(object):
    update_freq = 25.  # every 40 ms -- TODO check

    dtype = np.dtype('float')

    def __init__(self, channels):
        self.client = udp_feedback_client.Client()

    def start(self):
        self.client.start()
        self.data = self.client.get_feedback_data()

    def stop(self):
        self.client.stop()

    def get(self):
        d = self.data.next()

        return (d.state_name, np.array([d.value]))


# new
# use these 2 classes if using separate data sources for armassist and rehand

# for use with a DataSource
class ArmAssistData(object):
    update_freq = 25.  # every 40 ms -- TODO check

    state_names = ['aa_px', 'aa_py', 'aa_ppsi', 'aa_vx', 'aa_vy', 'aa_vpsi']
    dtype = np.dtype([(state_name, np.float64) for state_name in state_names])

    def __init__(self):
        self.client = udp_feedback_client.ArmAssistClient()

    def start(self):
        self.client.start()
        self.data = self.client.get_feedback_data()

    def stop(self):
        self.client.stop()

    def get(self):
        d = self.data.next()

        return np.array([tuple(d.data)], dtype=self.dtype)


# for use with a DataSource
class ReHandData(object):
    update_freq = 25.  # every 40 ms -- TODO check

    state_names = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono', 'rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']
    dtype = np.dtype([(state_name, np.float64) for state_name in state_names])

    def __init__(self):
        self.client = udp_feedback_client.ReHandClient()

    def start(self):
        self.client.start()
        self.data = self.client.get_feedback_data()

    def stop(self):
        self.client.stop()

    def get(self):
        d = self.data.next()

        return np.array([tuple(d.data)], dtype=self.dtype)
