'''
Code for getting data from kinarm
'''
import numpy as np
from .source import DataSourceSystem
import serial

class TouchData(DataSourceSystem):
    '''
    Client for data streamed from kinarm, compatible with riglib.source.DataSource
    '''
    update_freq = 1000.

    #  dtype is the numpy data type of items that will go 
    #  into the (multi-channel, in this case) datasource's ringbuffer
    dtype = np.dtype((np.float, (5,5)))

    def __init__(self):
        '''
            def __init__(self, addr=("192.168.0.8", 9090)):
        
        Constructor for Kinarmdata and connect to server

        Parameters
        ----------
        addr : tuple of length 2
            (client (self) IP address, client UDP port)

        
        self.conn = kinarmsocket.KinarmSocket(addr)
        self.conn.connect()
        '''
        self.conn = TouchDataInterface()
        self.conn.connect()

    def start(self):
        '''
        Start receiving data
        '''
        self.data = self.get_iterator()


    def get_iterator(self):
        assert self.conn.connected, "Socket is not connected, cannot get data"
        while self.conn.connected:
            self.conn.touch_port.write('t')
            tmp = self.conn.touch_port.read()
            tmp2 = np.zeros((5, 5))-1
            tmp2[0, 0] = float(tmp)
            yield tmp2

    def stop(self):
        '''
        Disconnect from kinarmdata socket
        '''
        self.conn.touch_port.close()

    def get(self):
        '''
        Get a new kinarm sample
        '''
        return next(self.data)

def make(cls=DataSourceSystem, *args, **kwargs):
    '''
    Docstring
    This ridiculous function dynamically creates a class with a new init function

    Parameters
    ----------

    Returns
    -------
    '''
    def init(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
    
    dtype = np.dtype((np.float, (1, )))    
    return type(cls.__name__, (cls,), dict(dtype=dtype, __init__=init))

class TouchDataInterface(object):
    def __init__(self):
        self.touch_port = serial.Serial('/dev/ttyACM2', baudrate=115200)

    def _test_connected(self):
        try:
            self.touch_port.write('t')
            tmp = int(self.touch_port.read())
            self.connected = True
        except:
            print('Error in interfacing w/ touch sensor')
            self.connected = False

    def connect(self):
        self._test_connected()
