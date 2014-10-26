'''Docstring.'''

import numpy as np

import udp_feedback_client


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

    # def get(self):
    #     d = self.data.next()
    #     return np.array([(tuple(d.data), tuple(d.ts), d.ts_sent, d.ts_arrival, d.freq)], dtype=self.dtype)

    # @staticmethod
    # def _get_dtype(state_names):
    #     sub_dtype_data = np.dtype([(name, np.float64) for name in state_names])
    #     sub_dtype_ts   = np.dtype([(name, np.int64)   for name in state_names])
    #     return np.dtype([('data',       sub_dtype_data),
    #                      ('ts',         sub_dtype_ts),
    #                      ('ts_sent',    np.float64),
    #                      ('ts_arrival', np.float64),
    #                      ('freq',       np.float64)])


class ArmAssistData(FeedbackData):
    '''For use with a DataSource in order to acquire feedback data from the 
    ArmAssist application.
    '''

    update_freq = 15.  # TODO check
    client_cls = udp_feedback_client.ArmAssistClient

    state_names = ['aa_px', 'aa_py', 'aa_ppsi']
    sub_dtype_data     = np.dtype([(name, np.float64) for name in state_names])
    sub_dtype_ts       = np.dtype([(name, np.int64)   for name in state_names])
    sub_dtype_data_aux = np.dtype([(name, np.float64) for name in ['force', 'bar_angle']])
    sub_dtype_ts_aux   = np.dtype([(name, np.int64)   for name in ['force', 'bar_angle']])
    
    dtype = np.dtype([('data',       sub_dtype_data),
                      ('ts',         sub_dtype_ts),
                      ('ts_arrival', np.int64),
                      ('data_aux',   sub_dtype_data_aux),
                      ('ts_aux',     sub_dtype_ts_aux)])

    def get(self):
        d = self.data.next()
        return np.array([(tuple(d.data), 
                          tuple(d.ts), 
                          d.ts_arrival, 
                          tuple(d.data_aux),
                          tuple(d.ts_aux))], 
                        dtype=self.dtype)


class ReHandData(FeedbackData):
    '''For use with a DataSource in order to acquire feedback data from the 
    ReHand application.
    '''

    update_freq = 200.  # TODO check
    client_cls = udp_feedback_client.ReHandClient

    state_names = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono', 
                   'rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']
    sub_dtype_data   = np.dtype([(name, np.float64) for name in state_names])
    sub_dtype_ts     = np.dtype([(name, np.int64)   for name in state_names])
    sub_dtype_torque = np.dtype([(name, np.float64) for name in ['thumb', 'index', 'fing3', 'prono']])
    
    dtype = np.dtype([('data',       sub_dtype_data),
                      ('ts',         sub_dtype_ts),
                      ('ts_arrival', np.int64),
                      ('freq',       np.float64),
                      ('torque',     sub_dtype_torque),
                      ('ts_sent',    np.int64)])

    def get(self):
        d = self.data.next()
        return np.array([(tuple(d.data), 
                          tuple(d.ts), 
                          d.ts_arrival, 
                          d.freq,
                          tuple(d.torque),
                          d.ts_sent)], 
                        dtype=self.dtype)

