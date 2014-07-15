'''Client-side code to receive feedback data from the ArmAssist and ReHand.'''

import sys
import time
import socket
import select
import numpy as np
from collections import namedtuple

from riglib.ismore import settings
from utils.constants import *

field_names = ['data', 'ts', 'ts_sent', 'ts_arrival', 'freq']
ArmAssistFeedbackData = namedtuple("ArmAssistFeedbackData", field_names)
ReHandFeedbackData    = namedtuple("ReHandFeedbackData",    field_names)


class Client(object):
    '''Docstring.'''

    MAX_MSG_LEN = 200

    # TODO -- rename this function to something else?
    def _create_and_bind_socket(self):
        '''Called in subclasses in their __init__() method.'''
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.address)

        self._init = True
        
    def start(self):
        self.listening = True

    def stop(self):
        self.listening = False
        self.file_.close()
    
    def __del__(self):
        self.stop()

    def get_feedback_data(self):
        raise NotImplementedError('Implement in subclasses!')


class ArmAssistClient(Client):
    '''Client code for receiving feedback data packets over UDP from the 
    ArmAssist application.'''

    address = settings.armassist_udp_client

    def __init__(self):
        self._create_and_bind_socket()

        self.file_ = open('armassist_feedback.txt', 'w')

    def get_feedback_data(self):
        '''Yield received feedback data.'''

        sleep_time = 0

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                ts_arrival = int(time.time() * 1e6)
                # print "feedback aa:", feedback
                self.sock.sendto("ACK ArmAssist\r", settings.armassist_udp_server)

                self.file_.write(feedback.rstrip('\r') + "\n")

                # Example feedback string:
                # "Status ArmAssist freq px py ppsi ts force bar_angle ts_aux\r"

                items = feedback.rstrip('\r').split(' ')
                
                cmd_id = items[0]
                dev_id = items[1]
                assert cmd_id == 'Status'
                assert dev_id == 'ArmAssist'

                freq = float(items[2])
                
                # position data and corresponding timestamp
                px   = float(items[3]) * mm_to_cm
                py   = float(items[4]) * mm_to_cm
                ppsi = float(items[5]) * deg_to_rad
                ts   = int(items[6])

                # print "ArmAssist timestamps:"
                # print "ts        ", ts
                # print "ts arrival", ts_arrival

                # auxiliary data and corresponding timestamp
                force     = float(items[7])
                bar_angle = float(items[8])
                ts_aux    = int(items[9])

                data = np.array([px, py, ppsi])
                ts   = np.array([ts, ts, ts])

                ts_sent = 0  # TODO -- fix

                yield ArmAssistFeedbackData(data=data,
                                            ts=ts,
                                            ts_sent=ts_sent,
                                            ts_arrival=ts_arrival,
                                            freq=freq)

            time.sleep(sleep_time)


class ReHandClient(Client):
    '''Client code for receiving feedback data packets over UDP from the 
    ReHand application.'''

    address = settings.rehand_udp_client

    def __init__(self):
        self._create_and_bind_socket()

        self.file_ = open('rehand_feedback.txt', 'w')

    def get_feedback_data(self):
        '''Yield received feedback data.'''

        sleep_time = 0

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                ts_arrival = int(time.time() * 1e6)  # microseconds
                #print "feedback rh:", feedback
                #self.sock.sendto("ACK ReHand\r", settings.rehand_udp_server)

                self.file_.write(feedback.rstrip('\r') + "\n")

                items = feedback.rstrip('\r').split(' ')
                
                dev_id = items[0]
                cmd_id = items[1]
                assert dev_id == 'ReHand'
                assert cmd_id == 'Status'               

                data_fields = items[2:]

                freq = float(data_fields[0])

                # values = [float(s) for s in items[3:]]
                # assert len(values) == 16

                vel    = [float(data_fields[i]) for i in [1, 5,  9, 13]]
                pos    = [float(data_fields[i]) for i in [2, 6, 10, 14]]
                torque = [float(data_fields[i]) for i in [3, 7, 11, 15]]
                ts     = [  int(data_fields[i]) for i in [4, 8, 12, 16]]

                ts_sent = int(data_fields[17])

                # print "timestamps:"
                # print "ts thumb  ", ts[0] 
                # print "ts index  ", ts[1]
                # print "ts fing3  ", ts[2]
                # print "ts prono  ", ts[3]
                # print "ts sent   ", ts_sent
                # print "ts arrival", int(ts_arrival * 1e6)

                data = np.array(pos + vel)
                ts   = np.array(ts + ts)

                # convert angular values from deg to rad (and deg/s to rad/s)
                data *= deg_to_rad

                yield ReHandFeedbackData(data=data, 
                                         ts=ts, 
                                         ts_sent=ts_sent,
                                         ts_arrival=ts_arrival,
                                         freq=freq)

            time.sleep(sleep_time)
