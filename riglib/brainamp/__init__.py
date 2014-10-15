'''Docstring.'''

import numpy as np

import rda


class EMGEEGEOG(object):
    '''For use with a MultiChanDataSource in order to acquire streaming EMG/EEG/EOG
    data from the BrainProducts BrainVision Recorder.
    '''

    update_freq = 1000.
    dtype = np.dtype('float')

    def __init__(self, recorder_ip):
        self.conn = rda.Connection(recorder_ip)
        self.conn.connect()

    def start(self):
        self.conn.start_data()
        self.data = self.conn.get_data()

    def stop(self):
        self.conn.stop_data()

    def get(self):
        d = self.data.next()
        return (d.chan, np.array([d.uV_value], dtype='float'))
