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

# ArmAssistFeedbackData = namedtuple("ArmAssistFeedbackData", ['aa_px', 'aa_py', 'aa_ppsi', 
#                                                      'aa_vx', 'aa_vy', 'aa_vpsi', 
#                                                      'arrival_ts'])
ArmAssistFeedbackData = namedtuple("ArmAssistFeedbackData", ['data', 'arrival_ts'])

# ReHandFeedbackData = namedtuple("ReHandFeedbackData", ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono', 
#                                                'rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono', 
#                                                'arrival_ts'])
ReHandFeedbackData = namedtuple("ReHandFeedbackData", ['data', 'arrival_ts'])


class Client(object):
    '''Docstring.'''

    MAX_MSG_LEN = 100

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
        '''TODO -- Docstring.'''

        sleep_time = 0 #0.005

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                arrival_ts = time.time()
                # print 'received feedback:', feedback

                # final structure of feedback packets is still TBD
                # for now, assume feedback packets are of the form, e.g., :
                #   "Feedback ReHand pos pos pos pos vel vel vel vel\r"
                #      or 
                #   "Feedback ArmAssist pos pos pos vel vel vel\r"

                items = feedback.rstrip('\r').split(' ')
                cmd_id = items[0]
                dev_id = items[1]
                values = [float(s) for s in items[2:]]

                # TODO -- don't hardcode state names below, get them from corresponding state space models

                # determine state names corresponding to the values
                if dev_id == 'ArmAssist':
                    state_names = ['aa_px', 'aa_py', 'aa_ppsi', 'aa_vx', 'aa_vy', 'aa_vpsi']
                elif dev_id == 'ReHand':
                    state_names = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono', 'rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']
                else:
                    raise Exception('Feedback data received from unknown device: ' + dev_id)
                 
                for state_name, value in zip(state_names, values):
                    # for angular values, convert from deg to rad (and deg/s to rad/s)
                    if state_name not in ['aa_px', 'aa_py', 'aa_vx', 'aa_vy']:
                        value *= deg_to_rad
                    yield PlantFeedbackData(state_name=state_name, value=value, arrival_ts=arrival_ts)

            time.sleep(sleep_time)


class ArmAssistClient(Client):
    '''Docstring.'''

    def __init__(self):
        super(ArmAssistClient, self).__init__('127.0.0.1', 5002)

    def get_feedback_data(self):
        '''TODO -- Docstring.'''

        sleep_time = 0.020 #0.005

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                arrival_ts = time.time()
                # print 'received feedback:', feedback

                # final structure of feedback packets is still TBD
                # for now, assume ArmAssist feedback packets are of the form, e.g., :
                #   "Feedback ArmAssist pos pos pos vel vel vel\r"

                items = feedback.rstrip('\r').split(' ')
                cmd_id = items[0]
                dev_id = items[1]
                data = np.array([float(s) for s in items[2:]])

                # convert angular values from deg to rad (and deg/s to rad/s)
                data[2] *= deg_to_rad  # aa_ppsi
                data[5] *= deg_to_rad  # aa_vpsi

                assert dev_id == 'ArmAssist'
                assert len(data) == 6

                yield ArmAssistFeedbackData(data=data, arrival_ts=arrival_ts)

            time.sleep(sleep_time)


class ReHandClient(Client):
    '''Docstring.'''

    def __init__(self):
        super(ReHandClient, self).__init__('127.0.0.1', 5003)

    def get_feedback_data(self):
        '''TODO -- Docstring.'''

        sleep_time = 0.020 #0.005

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                arrival_ts = time.time()
                # print 'received feedback:', feedback

                # final structure of feedback packets is still TBD
                # for now, assume ReHand feedback packets are of the form, e.g., :
                #   "Feedback ReHand pos pos pos pos vel vel vel vel\r"

                items = feedback.rstrip('\r').split(' ')
                cmd_id = items[0]
                dev_id = items[1]
                data = np.array([float(s) for s in items[2:]])

                # convert angular values from deg to rad (and deg/s to rad/s)
                data *= deg_to_rad

                assert dev_id == 'ReHand'
                assert len(data) == 8

                yield ReHandFeedbackData(data=data, arrival_ts=arrival_ts)

            time.sleep(sleep_time)
