'''Client-side code to receive feedback data from the ArmAssist and ReHand. 
See ArmAssist and ReHand command guides for more details on protocol of what 
data is sent over UDP.
'''

import sys
import time
import socket
import select
import numpy as np
from collections import namedtuple

from riglib.ismore import settings
from utils.constants import *


common_fields = ['data', 'ts', 'ts_arrival', 'freq']
aa_fields = common_fields + ['data_aux', 'ts_aux']
rh_fields = common_fields + ['torque']

ArmAssistFeedbackData = namedtuple("ArmAssistFeedbackData", aa_fields)
ReHandFeedbackData    = namedtuple("ReHandFeedbackData",    rh_fields)


class Client(object):
    '''Abstract base class, not meant to be instantiated.'''

    MAX_MSG_LEN = 300

    def __init__(self):
        self._create_and_bind_socket()
        # self.file_ = open(self.feedback_filename, 'w')

    def _create_and_bind_socket(self):
        '''Called in subclasses in their __init__() method.'''
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.address)

        self._init = True
        
    def start(self):
        self.listening = True

    def stop(self):
        self.listening = False
        # self.file_.close()
    
    def __del__(self):
        self.stop()

    def get_feedback_data(self):
        raise NotImplementedError('Implement in subclasses!')


class ArmAssistClient(Client):
    '''Client code for receiving feedback data packets over UDP from the 
    ArmAssist application.'''

    address = settings.armassist_udp_client
    feedback_filename = 'armassist_feedback.txt'

    def get_feedback_data(self):
        '''Yield received feedback data.'''

        sleep_time = 0

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                ts_arrival = time.time()  # secs
                
                # print "feedback aa:", feedback
                # self.sock.sendto("ACK ArmAssist\r", settings.armassist_udp_server)
                # self.file_.write(feedback.rstrip('\r') + "\n")

                # Example feedback string:
                # "Status ArmAssist freq px py ppsi ts force bar_angle ts_aux\r"
                command = "Status ArmAssist ts px py ppsi ts_aux force bar_angle -1\r"


                items = feedback.rstrip('\r').split(' ')
                
                cmd_id = items[0]
                dev_id = items[1]
                assert cmd_id == 'Status'
                assert dev_id == 'ArmAssist'
                
                data_fields = items[2:]
                assert len(data_fields) == 8

                freq = float(data_fields[0])

                # position data
                px   = float(data_fields[1]) * mm_to_cm        # convert to cm
                py   = float(data_fields[2]) * mm_to_cm        # convert to cm
                ppsi = float(data_fields[3]) * deg_to_rad      # convert to rad
                ts   = int(data_fields[4])   * us_to_s         # convert to sec
                
                # auxiliary data
                force     = float(data_fields[5])              # kg
                bar_angle = float(data_fields[6]) * deg_to_rad # convert to rad
                ts_aux    = int(data_fields[7])   * us_to_s    # convert to sec

                data     = np.array([px, py, ppsi])
                data_aux = np.array([force, bar_angle])

                yield ArmAssistFeedbackData(data=data,
                                            ts=ts,
                                            ts_arrival=ts_arrival,
                                            freq=freq,
                                            data_aux=data_aux,
                                            ts_aux=ts_aux)

            time.sleep(sleep_time)


class ReHandClient(Client):
    '''Client code for receiving feedback data packets over UDP from the 
    ReHand application.'''

    address = settings.rehand_udp_client
    feedback_filename = 'rehand_feedback.txt'

    def get_feedback_data(self):
        '''Yield received feedback data.'''

        sleep_time = 0

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                ts_arrival = time.time()  # secs
                
                # print "feedback rh:", feedback
                # self.sock.sendto("ACK ReHand\r", settings.rehand_udp_server)
                # self.file_.write(feedback.rstrip('\r') + "\n")

                items = feedback.rstrip('\r').split(' ')
                
                # feedback packet starts with "ReHand Status ...", as opposed 
                #   to "Status ArmAssist" for ArmAssist feedback packets
                dev_id = items[0]
                cmd_id = items[1]
                assert dev_id == 'ReHand'
                assert cmd_id == 'Status'               

                data_fields = items[2:]
                assert len(data_fields) == 18

                freq    = float(data_fields[0])
                vel     = [float(data_fields[i]) for i in [1, 5,  9, 13]]
                pos     = [float(data_fields[i]) for i in [2, 6, 10, 14]]
                torque  = [float(data_fields[i]) for i in [3, 7, 11, 15]]
                ts = int(data_fields[4]) * us_to_s  # convert to secs

                # convert angular values from deg to rad (and deg/s to rad/s)
                data = np.array(pos + vel) * deg_to_rad

                yield ReHandFeedbackData(data=data, 
                                         ts=ts,
                                         ts_arrival=ts_arrival,
                                         freq=freq,
                                         torque=torque)

            time.sleep(sleep_time)
