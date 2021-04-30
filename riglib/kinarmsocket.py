import socket
import struct
import time
import numpy as np

class KinarmSocket(object):
    ''' Method for receiving UDP packets from Kinarm. The kinarm must be running the Dexterit-E task 
    called 'PK_send_UDP_kinarm', or any task with a 'UDP Send Binary' block which is located in the 
    xPC target tab of the Simulink library. Note: The normal 'UDP Send' input will NOT work -- packets
    will not be sent since the Dexterit task computer does not receive them (the xPC machine does). 

    The socket port that packets are sent to is also configured in this UDP Send Binary block. 

    Data format is double --> binary byte packing with Byte alignment = 1 (From Matlab: "The byte alignment 
        field specifies how the data types are aligned. The possible values are: 1, 2, 4, and 8. The byte 
        alignment scheme is simple, and starts each element in the list of signals on a boundary specified 
        by the alignment relative to the start of the vector."")

    Packet received are 3 x 50 matrix. Refer to Dexterit Manual (on desktop of Dexterit task computer) for 
    what the rows / columns refer to (page 71/82).  
    '''

    def __init__(self, addr=('192.168.0.8', 9090)):
        ''' Self-IP address is: 192.168.0.8 set manually. Once fully migrating system over to BMI3d, 
        will need to adjust for new IPs'''

        #Set up UDP socket (specificed by socket.SOCK_DGRAM. TCP woudl use socket.SOCK_STREAM)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 

        #Free up port for use if you just ran task:
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        #Bind address
        self.socket.bind(addr)
        self.data_size = 1200 # 3 x 50 x 8 bytes (double)
        self.remote_ip = '192.168.0.2'

    def _test_connected(self):
        try:
            data, add = self.socket.recvfrom(1200)
            unpacked_ = struct.unpack('150d', data)
            assert len(unpacked_) == 150
            self.connected = True
        except:
            print('Make sure Kinarm Task is running - error in recieving packets')
            self.connected = False

    def connect(self):
        self._test_connected()

    def get_data(self):
        '''
        A generator which yields packets as they are received
        '''

        assert self.connected, "Socket is not connected, cannot get data"
        
        while self.connected:
            packet, address = self.socket.recvfrom(self.data_size)

            #Make sure packet is from correct address: 
            if address[0] == self.remote_ip:
                arrival_ts = time.time()
                data = np.array(struct.unpack('150d', packet))

                if data.shape[0] == 150:
                    #reshape data into 3 x 50
                    kindata = data.reshape(50,3).T
                    #kindata = np.hstack((kindata, np.zeros((3, 1))))
                    #kindata[:, -1] = arrival_ts
                    yield kindata

    def disconnect(self):
        self.socket.close()
        self.connected = False

    def __del__(self):
        self.disconnect()




