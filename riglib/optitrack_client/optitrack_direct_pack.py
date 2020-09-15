from riglib.optitrack_client.NatNetClient import NatNetClient as TestClient
import numpy as np
from multiprocessing import Process,Lock
import pickle

mutex = Lock()

class System(object):
    """
    this is is the dataSource interface for getting the mocap at BMI3D's reqeust
    compatible with DataSourceSystem
    uses data_array to keep track of the lastest buffer
    """
    rigidBodyCount = 1
    update_freq = 120 
    dtype = np.dtype((np.float, (rigidBodyCount, 6))) #6 degress of freedo
    def __init__(self):
        self.rigid_body_count = 1 #for now,only one rigid body

        self.test_client = TestClient()
        self.num_length = 10 # slots for buffer
        self.data_array = [None] * self.num_length
        self.rotation_buffer = [None] * self.num_length
    
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
            #save to rotation buffer list
            self.rotation_buffer.insert(0,position)
            self.rotation_buffer.pop()
        
 
    def start(self):
        self.test_client.newFrameListener = self.receiveNewFrame
        self.test_client.rigidBodyListener =self.receiveRigidBodyFrame
        self.test_client.run()
        print('Started the interface thread')
    
    def stop(self):
        pass
    
    def get(self):
        current_value = None
        rotation_value = None
        pos_rot = None

        with mutex:
            current_value = self.data_array[0]
            rotation_value = self.rotation_buffer[0]
        
        #return the latest saved data
        if (not current_value is None) and (not rotation_value is None):
            pos_rot = np.concatenate((np.asarray(current_value),np.asarray(rotation_value)))
        
        pos_rot = np.expand_dims(pos_rot, axis = 0)
        print(pos_rot.shape)
        return pos_rot #return that (x,y,z, rotation matrix)

class Simulation(System):
    '''
    this class does all the things except when the optitrack is not broadcasting data
    the get function starts to return random numbers
    '''
    update_freq = 60 #Hz

    def get(self):
        mag_fac = 10
        current_value = np.random.rand(self.rigidBodyCount, 6) * mag_fac
        current_value = np.expand_dims(current_value, axis = 0)
        return current_value
        

if __name__ == "__main__":
    s = System()
    s.start()
    s.get()