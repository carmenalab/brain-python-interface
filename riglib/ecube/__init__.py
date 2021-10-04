from .pyeCubeStream import eCubeStream
import numpy as np
import time
from .file import *
import aopy

'''
#to do list
where to calculate spikes?
'''

def multi_chan_generator(data_block, channels, downsample=1):
    for idx in range(data_block.shape[1]):
        yield (channels[idx], data_block[::downsample,idx]) # yield one channel at a time

from riglib.source import DataSourceSystem
class Broadband(DataSourceSystem):
    '''
    Wrapper class for pyecubestream compatible with using in DataSource for
    buffering neural data. Compatible with riglib.source.MultiChanDataSource
    '''
    # Required by DataSourceSystem: update_freq and dtype (see make() below)
    update_freq = 25000.
    dtype = np.dtype('float')

    def __init__(self, headstage=7, channels=[1]):
        '''
        Constructor for ecube.Broadband

        Inputs:
            headstages [int array]: list of each headstage (1-indexed)
            channels [tuple array]: channel range (start, stop) for each headstage (1-indexed)
        '''
        # Initialize the servernode-control connection
        self.conn = eCubeStream(debug=True)
        self.headstage = headstage
        self.channels = channels

    def start(self):

        # Add the requested headstage channels if they are available
        available = self.conn.listavailable()[0][self.headstage-1] # (headstages, analog, digital); hs are 1-indexed
        for ch in self.channels:
            if ch > available:
                raise RuntimeError('requested channel {} is not available ({} connected)'.format(
                    ch, available))
            else:
                self.conn.add(('Headstages', self.headstage, (ch, ch))) # add channels one at a time
        
        added = self.conn.listadded() # in debug mode this prints out the added channels

        # Start streaming
        self.conn.start()

        # Start with an empty generator
        self.gen = iter(())
    
    def stop(self):

        # Stop streaming
        if self.conn.stop():

            # Remove the added sources
            self.conn.remove(('Headstages', self.headstage))

        else:
            del self.conn # try to force the streaming to end by deleting the ecube connection object

    def get(self):
        '''data
        Retrieve a packet from the server
        '''
        try:
            return next(self.gen)
        except StopIteration:
            data_block = self.conn.get() # in the form of (time_stamp, data_source, data_content)
            self.gen = multi_chan_generator(data_block[2], self.channels)
            return next(self.gen)

class LFP(Broadband):
    '''
    Downsample the incoming data to 1000Hz. Compatible with riglib.source.MultiChanDataSource
    '''

    update_freq = 1000.

    def get(self):
        '''data
        Retrieve a packet from the server
        '''
        try:
            return next(self.gen)
        except StopIteration:
            data_block = self.conn.get() # in the form of (time_stamp, data_source, data_content)
            self.gen = multi_chan_generator(data_block[2], self.channels, downsample=25)
            return next(self.gen)

class File(DataSourceSystem):
    '''
    Wrapper class for pyecubestream compatible with using in DataSource for
    buffering neural data. Compatible with riglib.source.MultiChanDataSource
    '''
    # Required by DataSourceSystem: update_freq and dtype (see make() below)
    update_freq = 1000.
    chunksize = 728
    dtype = np.dtype('float')

    def __init__(self, ecube_bmi_filename, channels):

        # Open the file
        self.channels = channels
        zero_idx_channels = [ch-1 for ch in channels]
        self.file = aopy.data.load_ecube_data_chunked(ecube_bmi_filename, "Headstages", channels=zero_idx_channels, chunksize=self.chunksize)

    def get(self):
        '''data
        Read a "packet" worth of data from the file
        '''
        try:
            return next(self.gen)
        except (StopIteration, AttributeError):
            time.sleep(1./(25000/self.chunksize))
            try:
                data_block = next(self.file)
            except StopIteration:
                data_block = np.zeros((int(728/25),1))
            self.gen = multi_chan_generator(data_block, self.channels, downsample=25)
            return next(self.gen)

