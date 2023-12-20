'''
Base code for 'optitrack' feature, compatible with Optitrack motiontracker
'''
import threading
import time
import numpy as np
from ..source import DataSourceSystem
import socket
import re
import aopy

UDP_PORT = 11999

class System(DataSourceSystem):
    '''
    Optitrack DataSourceSystem collects motion tracking data via UDP packets using natnet depacketizer
    '''
    update_freq = 240 # This may not always be the case, but lower frequencies will still work, just waste space in the circular buffer
    dtype = np.dtype((np.float, (6,)))

    def start(self):
        '''
        Just set the callback function
        '''
        # Format is "frame_count&le_x&le_y&le_diam&re_x&re_y&re_diam"

        self.sock = socket.socket(socket.AF_INET, # Internet
                            socket.SOCK_DGRAM) # UDP
        self.sock.bind(("", UDP_PORT))   

    def stop(self):
        self.sock.close()
    
    def get(self):
        '''
        Main logic -- parse the motion tracking data into a defined datatype
        '''      
        data, addr = self.sock.recvfrom(1024) # buffer size is 1024 bytes
        items = data.decode('utf-8').split('&')
        frame_count = float(items[0]) # ignored

        # Careful with the x and y positions as they can be "Inf"
        try:
            le_x = float(items[1])
        except:
            le_x = np.nan
        try:
            le_y = float(items[2])
        except:
            le_y = np.nan
        try:
            re_x = float(items[4])
        except:
            re_x = np.nan
        try:
            re_y = float(items[5])
        except:
            re_y = np.nan
        
        le_diam = float(items[3])
        re_diam = float(items[6])

        # Pack into (1,6) array
        coords = np.array([le_x, le_y, re_x, re_y, le_diam, re_diam])
        coords = np.expand_dims(coords, axis=0)
        return coords


#################
# Simulated data
#################
class SimulatedEye(threading.Thread):
    '''
    This is not tested
    '''
    update_rate = 240

    def __init__(self, radius=(0.2,0.04,0.21,0.05,20,21), speed=(0.5,1,0.4,1.2,0.1, 0.1)):
        self.stime = time.time()
        self.radius = radius
        self.speed = speed
        self.frame_count = 0
        self.sock = socket.socket(socket.AF_INET, # Internet
                            socket.SOCK_DGRAM) # UDP

    def run_once(self):
        '''
        Fake some motion data
        '''
        time.sleep(1./240)
        ts = (time.time() - self.stime)
        coords = np.multiply(self.radius, np.cos(np.divide(ts, self.speed) * 2 * np.pi))
        self.send_data(coords)

    def send_data(self, coords):
        data = f"{self.frame_count}&{coords[0]}&{coords[1]}&{coords[2]}&{coords[3]}&{coords[4]}&{coords[5]}"
        self.sock.sendto(data.encode(), ("localhost", UDP_PORT))
        self.frame_count += 1

    def run(self):
        while True:
            time.sleep(1./self.update_rate)
            self.run_once()

########################
# Playback from csv file
########################

class PlaybackEye(SimulatedEye):
    '''
    This is not tested
    '''
    def __init__(self, data_dir, filename):
        self.data, metadata = aopy.data.load_bmi3d_hdf_table(data_dir, filename, 'task')

    def run_once(self):
        '''
        Fake some motion data
        '''
        coords = self.data['eyedata'][self.frame_count]
        self.send_data(coords)




