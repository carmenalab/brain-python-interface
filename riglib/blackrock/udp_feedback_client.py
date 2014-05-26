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
        '''TODO -- Docstring.'''

        sleep_time = 0 #0.005

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                arrival_ts = time.time()
                # print 'received feedback:', feedback

                # final structure of feedback packets is still TBD
                # for now, assume, feedback packets are of the form, e.g., :
                #   "Feedback ReHand pos pos pos vel vel vel\r"
                #      or 
                #   "Feedback ArmAssist pos pos pos pos vel vel vel vel\r"

                items = feedback.rstrip('\r').split(' ')
                cmd_id = items[0]
                dev_id = items[1]
                values = [float(s) for s in items[2:]]

                # TODO -- don't hardcode state names below, get them from corresponding state space models

                # determine state names corresponding to the values
                if dev_id == 'ArmAssist':
                    state_names = ['aa_px', 'aa_py', 'aa_ang_pz', 'aa_vx', 'aa_vy', 'aa_ang_vz']
                elif dev_id == 'ReHand':
                    state_names = ['rh_ang_px', 'rh_ang_py', 'rh_ang_pz', 'rh_ang_pw', 'rh_ang_vx', 'rh_ang_vy', 'rh_ang_vz', 'rh_ang_pw']
                else:
                    raise Exception('Feedback data received from unknown device: ' + dev_id)
                 
                for state_name, value in zip(state_names, values):
                    yield PlantFeedbackData(state_name=state_name, value=value, arrival_ts=arrival_ts)

            time.sleep(sleep_time)
