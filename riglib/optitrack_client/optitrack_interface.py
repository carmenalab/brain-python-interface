from .NatNetClient import NatNetClient as TestClient
import numpy as np
from multiprocessing import Process,Lock
import pickle

mutex = Lock()

class MotionData(object):
    """
    this is is the dataSource interface for getting the mocap at BMI3D's reqeust
    """

    def __init__(self, num_length):
        self.test_client = TestClient()
        #self.data_array = np.zeros(num_length)
        #self.data_array = np.zeros(num_length)
        self.data_array = [None] * num_length
        self.num_length = num_length
    
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