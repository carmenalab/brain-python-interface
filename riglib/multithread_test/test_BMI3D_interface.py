from .test_client import TestClient
import numpy as np
from multiprocessing import Process,Lock
import pickle

mutex = Lock()

class MotionData(object):


    def __init__(self, num_length):
        self.test_client = TestClient()
        #self.data_array = np.zeros(num_length)
        #self.data_array = np.zeros(num_length)
        self.data_array = [None] * num_length
        self.num_length = num_length
    
    
    def receive_data(self, data):
        #print( "Received data from client", data)
        rec_num  = data

        #self.data_array[2:] = self.data_array[1:]
        
        #make a running buffer
        with mutex:
            self.data_array.insert(0,rec_num)
            self.data_array.pop()
        
        #save data to a 
        

    def start(self):
        self.test_client.dataListener = self.receive_data
        self.test_client.run()
        print('Start the interface thread')
    
    def stop(self):
        pass
    
    def get(self):
        current_value = None
        with mutex:
            current_value = self.data_array[0]
        #return the latest saved data
        return current_value