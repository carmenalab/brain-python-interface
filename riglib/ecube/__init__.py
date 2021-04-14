from .pyeCubeStream import eCubeStream
import numpy as np
import time

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
        self.conn = eCubeStream()

        # Add the requested headstage channels if they are available
        available = self.conn.listavailable()[0] # (headstages, analog, digital)
        for idx in range(len(headstages)):
            if idx >= len(channels):
                raise ValueError('channels must be the same length as headstages')
            elif channels[idx][1] > available[headstages[idx]-1]:
                raise RuntimeError('requested channels are not available')
            else:
                self.conn.add(('Headstages', headstages[idx], channels[idx]))

    def start(self):
        self.conn.start()
    
    def stop(self):
        self.conn.stop()

    def get(self):
        '''
        Retrieve a packet from the server
        '''
        data_block = self.conn.get() # in the form of (time_stamp, data_source, data_content)
        timestamp = data_block[0]
        data = data_block[2]
        return np.array([timestamp, data], dtype=self.dtype)


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

    


