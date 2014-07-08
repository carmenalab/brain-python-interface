'''Client-side code to receive feedback data from the ArmAssist and ReHand.'''

import sys
import time
import socket
import select
import numpy as np
from collections import namedtuple

# CONSTANTS
rad_to_deg = 180 / np.pi
deg_to_rad = np.pi / 180

mm_to_cm = 0.1
cm_to_mm = 10.

ArmAssistFeedbackData = namedtuple("ArmAssistFeedbackData", ['data', 'ts', 'arrival_ts'])
ReHandFeedbackData    = namedtuple("ReHandFeedbackData",    ['data', 'ts', 'arrival_ts'])


class Client(object):
    '''Docstring.'''

    MAX_MSG_LEN = 200

    def __init__(self, ip, port):
        address = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(address)        

        self._init = True
        
    def start(self):
        self.listening = True

    def stop(self):
        self.listening = False
    
    def __del__(self):
        self.stop()

    def get_feedback_data(self):
        '''Implement in subclasses.'''
        raise NotImplementedError


class ArmAssistClient(Client):
    '''Docstring.'''

    def __init__(self):
        super(ArmAssistClient, self).__init__('127.0.0.1', 5002)

    def get_feedback_data(self):
        '''Docstring.'''

        sleep_time = 0 #0.020 #0.005

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                # self.sock.sendto("ACK\r", ('127.0.0.1', 5001))
                arrival_ts = time.time()
                # print 'received feedback:', feedback

                # Expected ArmAssist feedback packet structure, as decided
                # by Sid and David on 7/8/14:
                # "Status ArmAssist freq px py ppsi ts force bar_angle ts_aux\r"

                items = feedback.rstrip('\r').split(' ')
                
                cmd_id = items[0]
                dev_id = items[1]
                assert cmd_id == 'Status'
                assert dev_id == 'ArmAssist'

                freq = int(items[2])
                
                # position data and corresponding timestamp
                px   = float(items[3]) * mm_to_cm
                py   = float(items[4]) * mm_to_cm
                ppsi = float(items[5]) * deg_to_rad
                ts   = int(items[6])

                # auxiliary data and corresponding timestamp
                force     = float(items[7])
                bar_angle = float(items[8])
                ts_aux    = int(items[9])

                data = np.array([px, py, ppsi])
                ts   = np.array([ts, ts, ts])

                yield ArmAssistFeedbackData(data=data, ts=ts, arrival_ts=arrival_ts)

            time.sleep(sleep_time)


class ReHandClient(Client):
    '''Docstring.'''

    def __init__(self):
        super(ReHandClient, self).__init__('127.0.0.1', 5003)

    def get_feedback_data(self):
        '''TODO -- Docstring.'''

        sleep_time = 0 #0.020 #0.005

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                arrival_ts = time.time()
                # print 'received feedback:', feedback

                items = feedback.rstrip('\r').split(' ')
                cmd_id = items[0]
                dev_id = items[1]
                freq   = items[2]  # don't need this
                values = [float(s) for s in items[3:]]

                assert dev_id == 'ReHand'
                assert len(values) == 16

                vel    = [values[0], values[4], values[8], values[12]]
                pos    = [values[1], values[5], values[9], values[13]]
                torque = [values[2], values[6], values[10], values[14]]
                ts     = [values[3], values[7], values[11], values[15]]

                ts = [int(t) for t in ts]

                data = np.array(pos + vel)
                ts   = np.array(ts + ts)

                # convert angular values from deg to rad (and deg/s to rad/s)
                data *= deg_to_rad

                yield ReHandFeedbackData(data=data, ts=ts, arrival_ts=arrival_ts)

            time.sleep(sleep_time)
