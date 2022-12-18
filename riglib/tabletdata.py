'''
Code for getting data from kinarm
'''

import time
import numpy as np
from riglib.source import DataSourceSystem



class Tabletdata(DataSourceSystem):
    '''
    Client for data streamed from kinarm, compatible with riglib.source.DataSource
    '''
    update_freq = 200.

    #  dtype is the numpy data type of items that will go 
    #  into the (multi-channel, in this case) datasource's ringbuffer
    dtype = np.dtype((np.float, (2,)))

    def __init__(self):
        '''
        Constructor for Kinarmdata and connect to server

        Parameters
        ----------
        addr : tuple of length 2
            (client (self) IP address, client UDP port)

        '''
        from . import tabletstream
        self.conn = tabletstream.System()

    def start(self):
        '''
        Start receiving data
        '''
        self.data = self.conn.get()

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
    
    dtype = np.dtype((np.float, (2,)))    
    return type(cls.__name__, (cls,), dict(dtype=dtype, __init__=init))