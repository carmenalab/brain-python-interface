'''
Base code for 'optitrack' feature, compatible with Optitrack motiontracker
'''
import time
import numpy as np
from ..source import DataSourceSystem

class System(DataSourceSystem):
    '''
    Optitrack DataSourceSystem collects motion tracking data via UDP packets using natnet depacketizer
    '''
    update_freq = 240 # This may not always be the case, but lower frequencies will still work, just waste space in the circular buffer
    
    def __init__(self, client, feature="rigid body", n_features=1):
        '''
        Don't start the client in this DataSourceSystem object since then it won't be available for 
        commands elsewhere, i.e. start/stop recording
        '''
        self.client = client
        self.feature = feature # rigid body, skeleton, marker
        self.n_features = n_features
        self.rigid_bodies = []
        self.skeletons = []
        self.markers = []
        self.timing = []
    
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
        self.client.run_once(timeout=1)
        
        # Extract coordinates from feature
        coords = np.empty((self.n_features, 3))
        coords[:] = np.nan
        if self.feature == "rigid body":
            for i in range(np.min((self.n_features, len(self.rigid_bodies)))):
                if self.rigid_bodies[i].tracking_valid:
                    coords[i] = self.rigid_bodies[i].position
        elif self.feature == "marker":
            for i in range(np.min((self.n_features, len(self.markers)))):
                coords[i] = self.markers[i].position
        elif self.feature == "skeleton":
            raise NotImplementedError()
        else:
            raise AttributeError("Feature type unknown!")

        # For HDFWriter we need a dim 0
        coords = np.expand_dims(coords, axis=0)
        return coords
    
    def _update(self, rigid_bodies, skeletons, markers, timing):
        '''
        Callback for natnet client
        '''
        self.rigid_bodies = rigid_bodies
        self.skeletons = skeletons
        self.markers = markers
        self.timing = timing


#################
# Simulated data
#################
class RigidBody():

    position = None
    tracking_valid = True
    def __init__(self, position):
        self.position = position

class SimulatedClient():

    def __init__(self, n=1, radius=(0.2,0.04,0.1), speed=(0.5,1,2)):
        self.stime = time.time()
        self.n = n
        self.radius = radius
        self.speed = speed

    def set_callback(self, callback):
        self.callback = callback

    def run_once(self, timeout=None):
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

########################
# Playback from csv file
########################

class PlaybackClient(SimulatedClient):

    def __init__(self, filename):
        import pandas as pd
        self.stime = time.time()
        csv = pd.read_csv(filename, header=[1,4,5])
        self.motiondata = csv['Rigid Body']['Position']
        self.time = csv['Type'].iloc[:,0]

    def run_once(self, timeout=None):
        '''
        Read one line of motion data from the csv file
        '''
        read_freq = 240 # doesn't really matter if we read too fast... 
        time.sleep(1./read_freq)
        ts = (time.time() - self.stime)
        coords = np.empty((3,))
        coords[:] = np.nan
        now = (i for i,t in enumerate(self.time) if t > ts) # ...because we check the timestamps here
        try:
            row = next(now)
            coords[0] = self.motiondata.iloc[row].X
            coords[1] = self.motiondata.iloc[row].Y
            coords[2] = self.motiondata.iloc[row].Z
        except:
            pass
        data = [RigidBody(coords)]
        self.callback(data, [], [], [])

# System definition function
def make(cls, client, feature, num_features=1, **kwargs):
    """
    This ridiculous function dynamically creates a class with a new init function
    """
    def init(self):
        super(self.__class__, self).__init__(client, feature, num_features, **kwargs)
    
    dtype = np.dtype((np.float, (num_features, 3)))
    return type(cls.__name__, (cls,), dict(dtype=dtype, __init__=init))