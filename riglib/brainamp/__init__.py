# '''Docstring.'''

# import numpy as np

# import rda


# class EMG(object):
#     '''For use with a MultiChanDataSource in order to acquire streaming EMG/EEG/EOG
#     data (not limited to just EEG) from the BrainProducts BrainVision Recorder.
#     '''

#     update_freq = 2500.  # TODO -- check
#     dtype = np.dtype('float')

#     # TODO -- added **kwargs argument to __init__ for now because MCDS is passing
#     #   in source_kwargs which contains 'channels' kwarg which is not needed/expected
#     #   need to fix this later
#     def __init__(self, recorder_ip='192.168.137.1', **kwargs):
#         print '\ninside brainamp.__init__.EMG'
#         print 'recorder_ip', recorder_ip
#         self.conn = rda.Connection(recorder_ip)
#         self.conn.connect()

#     def start(self):
#         self.conn.start_data()
#         self.data = self.conn.get_data()

#     def stop(self):
#         self.conn.stop_data()

#     def get(self):
#         d = self.data.next()
#         return (d.chan, np.array([d.uV_value], dtype='float'))
