'''Client-side code to receive feedback data from the ArmAssist and ReHand.'''

import sys
import time
import socket
import select
from collections import namedtuple

PlantFeedbackData = namedtuple("PlantFeedbackData", ["state_name", "value", "arrival_ts"])

class Client(object):
    '''Docstring.'''

    MAX_MSG_LEN = 100

    def __init__(self):
        ip = '127.0.0.1'
        port = 5002
        address = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(address)        

        self._init = True
        
    def start(self):
        '''Docstring.'''

        self.listening = True

    def stop(self):
        '''Docstring.'''

        self.listening = False
    
    def __del__(self):
        self.stop()

    def get_feedback_data(self):
        '''Docstring.'''

        sleep_time = 0 #0.005

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                arrival_ts = time.time()

                # temporary for now
                # structure of feedback packets is still TBD
                state_names = ['aa_px', 'aa_py', 'aa_ang_pz', 'aa_vx', 'aa_vy', 'aa_ang_vz']
                values = [float(s) for s in feedback.rstrip('\r').split(' ')]
                
                for state_name, value in zip(state_names, values):
                    # print 'yielding'
                    yield PlantFeedbackData(state_name=state_name, value=value, arrival_ts=arrival_ts)

                time.sleep(sleep_time)
