from .NatNetClient import NatNetClient as TestClient
import numpy as np
from multiprocessing import Process,Lock
import pickle

mutex = Lock()

class System(object):
    """
    this is is the dataSource interface for getting the mocap at BMI3D's reqeust
    compatible with DataSourceSystem
    """
    rigidBodyCount = 1
    dtype = np.dtype((np.float, (rigidBodyCount, 6))) #6 degress of freedo
    def __init__(self):
        self.update_freq = 120 #Hz
        self.rigid_body_count = 1 #for now,only one rigid body

        self.test_client = TestClient()
        self.num_length = 10 # slots for buffer
        self.data_array = [None] * self.num_length
    
        # This is a callback function that gets connected to the NatNet client and called once per mocap frame.
    def receiveNewFrame(self, frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                        labeledMarkerCount, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged ):
        #print( "Received frame", frameNumber )
        pass

    # This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
    def receiveRigidBodyFrame(self, id, position, rotation ):
        #print( "Received frame for rigid body", position )

        #save to the running buffer with a lock
        with mutex:
            self.data_array.insert(0,position)
            self.data_array.pop()
        
 
    def start(self):
        self.test_client.newFrameListener = self.receiveNewFrame
        self.test_client.rigidBodyListener =self.receiveRigidBodyFrame
        self.test_client.run()
        print('Started the interface thread')
    
    def stop(self):
        pass
    
    def get(self):
        current_value = None
        with mutex:
            current_value = self.data_array[0]
        #return the latest saved data
        return current_value

class Simulation(System):
    '''
    this class does all the things except when the optitrack is not broadcasting data
    the get function starts to return random numbers
    '''
    update_freq = 10 #Hz

    def get(self):
        mag_fac = 10
        current_value = np.random.rand(self.rigidBodyCount, 6) * mag_fac
        return current_value
        