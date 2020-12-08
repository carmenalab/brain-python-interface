'''
Base code for 'optitrack' feature, compatible with Optitrack motiontracker
'''

import os
import time
import numpy as np

class System(object):
    '''
    Optitrack DataSourceSystem runs at 240 Hz collects data via UDP packets using natnet depacketizer
    '''
    update_freq = 240
    
    def __init__(self, client, feature="rigid body", n_features=1):
        '''
        Don't start the client in this DataSourceSystem object since then it won't be available for 
        commands elsewhere, i.e. start/stop recording
        '''
        self.client = client
        self.feature = feature # rigid body, skeleton, marker
        self.n_features = n_features
    
    def start(self):
        '''
        Just set the callback function
        '''
        self.client.set_callback(
            lambda rb, s, m, t: self._update(rb, s, m, t))

    def stop(self):
        pass
    
    def get(self):
        '''
        Main logic -- parse the motion tracking data into a defined datatype
        '''

        # Run the client to collect a frame of data
        self.client.run_once()

        # Pick a feature
        if self.feature == "rigid body":
            feature = self.rigid_bodies
        elif self.feature == "skeleton":
            feature = self.skeletons
        elif self.feature == "marker":
            feature = self.markers
        else:
            raise AttributeError("Feature type unknown!")
        
        # Extract coordinates from feature
        coords = np.zeros((self.n_features, 3))
        for i in range(np.min((self.n_features, len(feature)))):
            coords[i] = feature[i].position
        return coords
    
    def _update(self, rigid_bodies, skeletons, markers, timing):
        '''
        Callback for natnet client
        '''
        self.rigid_bodies = rigid_bodies
        self.skeletons = skeletons
        self.markers = markers
        self.timing = timing

class RigidBody():

    position = None
    def __init__(self, position):
        self.position = position

class SimulatedClient():

    def __init__(self, n=1, radius=(20,4,10), speed=(5,1,2)):
        self.stime = time.time()
        self.n = n
        self.radius = radius
        self.speed = speed

    def set_callback(self, callback):
        self.callback = callback

    def run_once(self):
        '''
        Fake some motion data
        '''
        time.sleep(1./240)
        ts = (time.time() - self.stime)
        coords = np.multiply(self.radius, np.cos(np.divide(ts, self.speed) * 2 * np.pi))
        data = [RigidBody(coords)]
        self.callback(data, [], [], [])

    def start_recording(self):
        print("Start recording")

    def stop_recording(self):
        print("Stop recording")

    def set_take(self, take_name):
        print("Setting take_name: " + take_name)

    def set_session(self, session_name):
        print("Setting session_name: " + session_name)

def make(cls, client, optitrack_feature, optitrack_num_features, **kwargs):
    """
    This ridiculous function dynamically creates a class with a new init function
    """
    def init(self):
        super(self.__class__, self).__init__(client, optitrack_feature, optitrack_num_features, **kwargs)
    
    dtype = np.dtype((np.float, (optitrack_num_features, 3)))
    return type(cls.__name__, (cls,), dict(dtype=dtype, __init__=init))