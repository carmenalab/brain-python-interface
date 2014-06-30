import numpy as np
import rda

class EMG(object):
    update_freq = 2500.  # TODO -- double check

    dtype = np.dtype('float')

    def __init__(self, channels):
        recorder_ip_addr = '192.168.137.1'  # TODO
        self.conn = rda.Connection(recorder_ip_addr)
        self.conn.connect()

    def start(self):
        self.conn.start_data()
        self.data = self.conn.get_data()

    def stop(self):
        self.conn.stop_data()

    def get(self):
        d = self.data.next()

        return (d.chan, np.array([d.uV_value], dtype='float'))
