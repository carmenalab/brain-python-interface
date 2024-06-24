import socket
import struct
from threading import Thread
import pickle
import numpy

class TestClient:
    def __init__(self):
        # Change this value to the IP address of the NatNet server.
        self.serverIPAddress = "127.0.0.1" 

        # Change this value to the IP address of your local network interface
        self.localIPAddress = "127.0.0.1"

        self.dataPort = 5005

        # similar to the rigidBodyListener 
        self.dataListener = None

        # Create a data socket to attach to the NatNet stream
    def __createDataSocket( self, port ):
        result = socket.socket( socket.AF_INET,     # Internet
                              socket.SOCK_DGRAM,
                              socket.IPPROTO_UDP)    # UDP

        #result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)        
        #result.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(self.multicastAddress) + socket.inet_aton(self.localIPAddress))

        result.bind( (self.localIPAddress, port) )

        return result

    def __dataThreadFunction( self, socket ):
        while True:
            # Block for input
            data, addr = socket.recvfrom( 32768 ) # 32k byte buffer size
            if( len( data ) > 0 ):
                #self.__processMessage( data )
                data_arr  = numpy.asarray(pickle.loads(data))
                # Send information to any listener.
                if self.dataListener is not None:
                    self.dataListener(data_arr)

    def run( self ):
        # Create the data socket
        self.dataSocket = self.__createDataSocket( self.dataPort )
        if( self.dataSocket is None ):
            print( "Could not open data channel" )
            exit

        # Create a separate thread for receiving data packets
        dataThread = Thread( target = self.__dataThreadFunction, args = (self.dataSocket, ))
        dataThread.start()