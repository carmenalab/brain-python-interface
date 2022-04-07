'''
Code for getting data from kinarm
'''

import time
import numpy as np
from .source import DataSourceSystem
from . import kinarmsocket

class Kinarmdata(DataSourceSystem):
    '''
    Client for data streamed from kinarm, compatible with riglib.source.DataSource
    '''
    update_freq = 1000.

    #  dtype is the numpy data type of items that will go 
    #  into the (multi-channel, in this case) datasource's ringbuffer
    dtype = np.dtype((np.float, (3, 50)))

    def __init__(self, addr=("192.168.0.8", 9090)):
        '''
        Constructor for Kinarmdata and connect to server

        Parameters
        ----------
        addr : tuple of length 2
            (client (self) IP address, client UDP port)

        '''
        self.conn = kinarmsocket.KinarmSocket(addr)
        self.conn.connect()

    def start(self):
        '''
        Start receiving data
        '''
        self.data = self.conn.get_data()

    def stop(self):
        '''
        Disconnect from kinarmdata socket
        '''
        self.conn.disconnect()

    def get(self):
        '''
        Get a new kinarm sample
        '''
        # while True:
            # try:
            #     d = self.data.next()
            # except:
            #     break

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
    
    dtype = np.dtype((np.float, (3, 50)))    
    return type(cls.__name__, (cls,), dict(dtype=dtype, __init__=init))