from .pyeCubeStream import eCubeStream
import numpy as np
import time
from .file import *

'''
#to do list
where to calculate spikes?
'''

from riglib.source import DataSourceSystem
class Broadband(DataSourceSystem):
    '''
    Wrapper class for pyecubestream compatible with using in DataSource for
    buffering neural data.
    '''
    # Required by DataSourceSystem: update_freq and dtype (see make() below)
    update_freq = 25000/728.

    def __init__(self, headstages=[7], channels=[(1, 640)]):
        '''
        Constructor for ecube.Broadband

        Inputs:
            headstages [int array]: list of each headstage (1-indexed)
            channels [tuple array]: channel range (start, stop) for each headstage (1-indexed)
        '''
        # Initialize the servernode-control connection
        self.conn = eCubeStream(debug=True)
        self.headstages = headstages
        self.channels = channels

    def start(self):

        # Add the requested headstage channels if they are available
        available = self.conn.listavailable()[0] # (headstages, analog, digital)
        for idx in range(len(self.headstages)):
            if idx >= len(self.channels):
                raise ValueError('channels must be the same length as headstages')
            elif self.channels[idx][1] > available[self.headstages[idx]-1]: # hs are 1-indexed
                raise RuntimeError('requested channels {} are not available ({} available)'.format(
                    self.channels[idx], available[self.headstages[idx]]))
            else:
                self.conn.add(('Headstages', self.headstages[idx], self.channels[idx]))
        
        added = self.conn.listadded() # in debug mode this prints out the added channels

        # Start streaming
        self.conn.start()
    
    def stop(self):

        # Stop streaming
        self.conn.stop()

        # Remove the added sources
        for idx in range(len(self.headstages)):
            self.conn.remove(('Headstages', self.headstages[idx]))

    def get(self):
        '''data
        Retrieve a packet from the server
        '''
        data_block = self.conn.get() # in the form of (time_stamp, data_source, data_content)
        return_value = np.empty((1,), dtype=self.dtype)
        return_value['timestamp'] = data_block[0]
        return_value['data'] = data_block[2]
        return return_value

# System definition function
def make(cls, headstages=[7], channels=[(1, 640)], **kwargs):
    """
    This ridiculous function dynamically creates a class with a new init function
    """
    def init(self):
        super(self.__class__, self).__init__(headstages, channels, **kwargs)
    
    # Sum over all the channels to know how big to make the buffer
    nch = int(np.sum([1+ch[1]-ch[0] for ch in channels]))
    dtype = np.dtype([('timestamp', 'u8'), ('data', 'i4', (728,nch))])
    return type(cls.__name__, (cls,), dict(dtype=dtype, __init__=init))
