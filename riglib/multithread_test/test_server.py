import socket,time
from threading import Thread
import numpy as np
import string
import pickle

import pyautogui



'''
this class generates random numbers and broadcast them via the UDP protocal to the local network
print "UDP target IP:", UDP_IP
print "UDP target port:", UDP_PORT
print "message:", MESSAGE
'''
class TestServer(object): 
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    MESSAGE = "Hello, World!"
    SLEEP_TIME = 0.01 # seond 100 Hz per second

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
    
    def __dataThreadFunction( self, socket,sleep_time ):
        while True:
            # Block for input
            rand_num = np.random.rand()

            self.sock.sendto(str(rand_num).encode(), (self.UDP_IP, self.UDP_PORT))
            time.sleep(sleep_time)
    
    def run( self ):
        # Create a separate thread for receiving data packets
        dataThread = Thread( target = self.__dataThreadFunction, args = (self.sock, self.SLEEP_TIME))
        print('Server starts to broadcast data')
        dataThread.start() 

#this child class replaces the generator and then waits for the mouse command
class TestServerMouse(TestServer):

    def __init__(self):
        
        super().__init__()

    def __dataThreadFunction( self, socket,sleep_time ):

        while True:
            #get cursor position with pyautogui
            cursor_pos = pyautogui.position()

            #prepare dump data
            dump_data = pickle.dumps(cursor_pos)

            self.sock.sendto(dump_data, (self.UDP_IP, self.UDP_PORT))
            time.sleep(sleep_time)
    
    def run(self):
        dataThread = Thread( target = self.__dataThreadFunction, args = (self.sock, self.SLEEP_TIME))
        print('Server starts to broadcast data')
        dataThread.start() 
        
        

#test function 
if __name__ == "__main__":
    tsm = TestServerMouse()
    tsm.run()