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

PlantFeedbackData = namedtuple("PlantFeedbackData", ["state_name", "value", "arrival_ts"])

ArmAssistFeedbackData = namedtuple("ArmAssistFeedbackData", ['data', 'ts', 'arrival_ts'])
ReHandFeedbackData = namedtuple("ReHandFeedbackData", ['data', 'ts', 'arrival_ts'])


class Client(object):
    '''Docstring.'''

    MAX_MSG_LEN = 200

    def __init__(self, ip='127.0.0.1', port=5002):
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
        raise NotImplementedError
    #     '''TODO -- Docstring.'''

    #     sleep_time = 0 #0.005

    #     while self.listening:
    #         r, _, _ = select.select([self.sock], [], [], 0)
            
    #         if r:  # if the list r is not empty
    #             feedback = self.sock.recv(self.MAX_MSG_LEN)
    #             arrival_ts = time.time()
    #             # print 'received feedback:', feedback

    #             items = feedback.rstrip('\r').split(' ')
    #             cmd_id = items[0]
    #             dev_id = items[1]
    #             freq   = items[2]  # don't need this
    #             values = [float(s) for s in items[3:]]

    #             vel = []
    #             pos = []
    #             torque = []

    #             i = 0
    #             for i, value in enumerate(values):
    #                 if i % 3 == 0:
    #                     vel.append(value)
    #                 if i % 3 == 1:
    #                     pos.append(value)
    #                 if i % 3 == 2:
    #                     torque.append(value)

    #             # determine state names corresponding to the values
    #             if dev_id == 'ArmAssist':
    #                 state_names = ['aa_px', 'aa_py', 'aa_ppsi', 'aa_vx', 'aa_vy', 'aa_vpsi']
    #             elif dev_id == 'ReHand':
    #                 state_names = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono', 'rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']
    #             else:
    #                 raise Exception('Feedback data received from unknown device: ' + dev_id)
                 
    #             for state_name, value in zip(state_names, pos + vel):
    #                 # for angular values, convert from deg to rad (and deg/s to rad/s)
    #                 if state_name not in ['aa_px', 'aa_py', 'aa_vx', 'aa_vy']:
    #                     value *= deg_to_rad
    #                 yield PlantFeedbackData(state_name=state_name, value=value, arrival_ts=arrival_ts)

    #         time.sleep(sleep_time)


class ArmAssistClient(Client):
    '''Docstring.'''

    def __init__(self):
        super(ArmAssistClient, self).__init__('127.0.0.1', 5002)

    def get_feedback_data(self):
        '''TODO -- Docstring.'''

        sleep_time = 0 #0.020 #0.005

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                # self.sock.sendto("ACK\r", ('127.0.0.1', 5001))
                arrival_ts = time.time()
                # print 'received feedback:', feedback

                items = feedback.rstrip('\r').split(' ')
                cmd_id = items[0]
                dev_id = items[1]
                freq   = items[2]  # don't need this
                values = [float(s) for s in items[3:]]

                assert dev_id == 'ArmAssist'
                assert len(values) == 12

                vel    = [values[0], values[4], values[8]]
                pos    = [values[1], values[5], values[9]]
                torque = [values[2], values[6], values[10]]
                ts     = [values[3], values[7], values[11]]

                ts = [int(t) for t in ts]
                
                data = np.array(pos + vel)
                ts   = np.array(ts + ts)

                # convert angular values from deg to rad (and deg/s to rad/s)
                data[2] *= deg_to_rad  # aa_ppsi
                data[5] *= deg_to_rad  # aa_vpsi                

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
