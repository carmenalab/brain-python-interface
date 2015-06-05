'''Client-side code to receive feedback data from the ArmAssist and ReHand. 
See ArmAssist and ReHand command guides for more details on protocol of what 
data is sent over UDP.
'''

import sys
import time
import socket
import select
import numpy as np

from riglib.ismore import settings
from utils.constants import *


class FeedbackData(object):
    '''Abstract base class, not meant to be instantiated.'''

    MAX_MSG_LEN = 300
    sleep_time = 0

    # must define these in subclasses
    update_freq = None
    address     = None
    dtype       = None

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.address)

        # self.file_ = open(self.feedback_filename, 'w')

    def start(self):
        self.listening = True
        self.data = self.get_feedback_data()

    def stop(self):
        self.listening = False
        self.sock.close()
        # self.file_.close()

    # TODO -- is this even necessary?
    def __del__(self):
        self.stop()

    # TODO -- add comment about how this will get called by the source
    def get(self):
        return self.data.next()

    def get_feedback_data(self):
        '''Yield received feedback data.'''

        self.last_timestamp = -1

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                ts_arrival = time.time()  # secs
                
                # print "feedback:", feedback
                # self.file_.write(feedback.rstrip('\r') + "\n")

                processed_feedback = self.process_received_feedback(feedback, ts_arrival)

                if processed_feedback['ts'] != self.last_timestamp:
                    yield processed_feedback

                self.last_timestamp = processed_feedback['ts']

            time.sleep(self.sleep_time)

    def process_received_feedback(self, feedback, ts_arrival):
        raise NotImplementedError('Implement in subclasses!')
            

class ArmAssistData(FeedbackData):
    '''Client code for use with a DataSource in order to acquire feedback data over UDP from the 
    ArmAssist application.
    '''

    update_freq = 25.
    address     = settings.ARMASSIST_UDP_CLIENT_ADDR
    #feedback_filename = 'armassist_feedback.txt'

    state_names = ['aa_px', 'aa_py', 'aa_ppsi']

    sub_dtype_data     = np.dtype([(name, np.float64) for name in state_names])
    sub_dtype_data_aux = np.dtype([(name, np.float64) for name in ['force', 'bar_angle']])
    
    dtype = np.dtype([('data',       sub_dtype_data),
                      ('ts',         np.float64),
                      ('ts_arrival', np.float64),
                      ('freq',       np.float64),
                      ('data_aux',   sub_dtype_data_aux),
                      ('ts_aux',     np.float64)])

    def process_received_feedback(self, feedback, ts_arrival):
        '''Process feedback strings of the form:
            "Status ArmAssist freq px py ppsi ts force bar_angle ts_aux\r"
        '''

        items = feedback.rstrip('\r').split(' ')
        
        cmd_id      = items[0]
        dev_id      = items[1]
        data_fields = items[2:]
        
        assert cmd_id == 'Status'
        assert dev_id == 'ArmAssist'
        assert len(data_fields) == 8

        freq = float(data_fields[0])                    # Hz

        # position data
        px   = float(data_fields[1]) * mm_to_cm         # cm
        py   = float(data_fields[2]) * mm_to_cm         # cm
        ppsi = float(data_fields[3]) * deg_to_rad       # rad
        ts   = int(data_fields[4])   * us_to_s          # sec
        
        # auxiliary data
        force     = float(data_fields[5])               # kg
        bar_angle = float(data_fields[6]) * deg_to_rad  # rad
        ts_aux    = int(data_fields[7])   * us_to_s     # sec

        data     = (px, py, ppsi)
        data_aux = (force, bar_angle)

        return np.array([(data,
                          ts,
                          ts_arrival,
                          freq,
                          data_aux,
                          ts_aux)],
                        dtype=self.dtype)


class ReHandData(FeedbackData):
    '''Client code for use with a DataSource in order to acquire feedback data over UDP from the 
    ReHand application.
    '''

    update_freq = 200.
    address     = settings.REHAND_UDP_CLIENT_ADDR
    #feedback_filename = 'rehand_feedback.txt'

    state_names = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono', 
                   'rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']
    sub_dtype_data   = np.dtype([(name, np.float64) for name in state_names])
    sub_dtype_torque = np.dtype([(name, np.float64) for name in ['thumb', 'index', 'fing3', 'prono']])
    
    dtype = np.dtype([('data',       sub_dtype_data),
                      ('ts',         np.float64),
                      ('ts_arrival', np.float64),
                      ('freq',       np.float64),
                      ('torque',     sub_dtype_torque)])

    def process_received_feedback(self, feedback, ts_arrival):
        '''Process feedback strings of the form:
            "ReHand Status freq vthumb pthumb tthumb ... tprono ts\r"
        '''

        items = feedback.rstrip('\r').split(' ')
                
        # feedback packet starts with "ReHand Status ...", as opposed 
        #   to "Status ArmAssist ... " for ArmAssist feedback packets
        dev_id      = items[0]
        cmd_id      = items[1]
        data_fields = items[2:]

        assert dev_id == 'ReHand'
        assert cmd_id == 'Status'
        assert len(data_fields) == 14

        freq = float(data_fields[0])

        #display data before being converted to radians
        '''
        print "thumb float:", float(data_fields[1])
        print "thumb :", (data_fields[1])
        print "index float:", float(data_fields[4])
        print "index:", float(data_fields[4])
        print "3fing float:", float(data_fields[7])
        print "3fing :", float(data_fields[7])
        print "prono float:", float(data_fields[10])
        print "prono:", float(data_fields[10])

        '''

        # velocity, position, and torque for the 4 ReHand joints
        vthumb = float(data_fields[1])  * deg_to_rad  # rad
        pthumb = float(data_fields[2])  * deg_to_rad  # rad
        tthumb = float(data_fields[3])                # mNm
        vindex = float(data_fields[4])  * deg_to_rad  # rad
        pindex = float(data_fields[5])  * deg_to_rad  # rad
        tindex = float(data_fields[6])                # mNm
        vfing3 = float(data_fields[7])  * deg_to_rad  # rad
        pfing3 = float(data_fields[8])  * deg_to_rad  # rad
        tfing3 = float(data_fields[9])                # mNm
        vprono = float(data_fields[10]) * deg_to_rad  # rad
        pprono = float(data_fields[11]) * deg_to_rad  # rad
        tprono = float(data_fields[12])               # mNm



        ts = int(data_fields[13]) * us_to_s           # secs

        data   = (pthumb, pindex, pfing3, pprono,
                  vthumb, vindex, vfing3, vprono)
        torque = (tthumb, tindex, tfing3, tprono)

        return np.array([(data,
                          ts, 
                          ts_arrival, 
                          freq,
                          torque)],
                        dtype=self.dtype)
