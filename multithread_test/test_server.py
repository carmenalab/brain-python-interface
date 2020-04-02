import socket,time
from threading import Thread
import numpy as np
import string



'''
print "UDP target IP:", UDP_IP
print "UDP target port:", UDP_PORT
print "message:", MESSAGE
'''
class TestServer(object):
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    MESSAGE = "Hello, World!"
    SLEEP_TIME = 0.01 # seond

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
