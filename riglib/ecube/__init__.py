from .pyeCubeStream import eCubeStream
import numpy as np
import time
from .file import *
import aopy
from riglib.source import DataSourceSystem

'''
eCube streaming sources
'''

def multi_chan_generator(data_block, channels, downsample=1):
    for idx in range(len(channels)):
        yield (channels[idx], data_block[::downsample,idx]) # yield one channel at a time

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
            headstages int: headstage number (1-indexed)
            channels [int array]: channel list (1-indexed)
        '''
        # Initialize the servernode-control connection
        self.conn = eCubeStream(debug=False)
        self.headstage = headstage
        self.channels = channels

    def start(self):
        print("Starting ecube streaming datasource...")

        # Remove all existing sources
        subscribed = self.conn.listadded()
        if len(subscribed[0]) > 0:
            self.conn.remove(('Headstages', self.headstage))
        if len(subscribed[1]) > 0:
            self.conn.remove(('AnalogPanel',))
        if len(subscribed[2]) > 0:
            self.conn.remove(('DigitalPanel',))

        # Add the requested headstage channels if they are available
        available = self.conn.listavailable()[0][self.headstage-1] # (headstages, analog, digital); hs are 1-indexed
        for ch in self.channels:
            if ch > available:
                raise RuntimeError('requested channel {} is not available ({} connected)'.format(
                    ch, available))
            self.conn.add(('Headstages', self.headstage, (ch, ch))) # add channels one at a time
        subscribed = self.conn.listadded() # in debug mode this prints out the added channels

        # Start streaming
        self.conn.start()

        # Start with an empty generator
        self.gen = iter(())

        # Call get once to make sure we have the right channels
        data_block = self.conn.get()
        assert data_block[1] == "Headstages"
        assert data_block[2].shape[1] == len(self.channels)
        print(f"Started streaming from ecube, packet size {data_block[2].shape[0]}")
    
    def stop(self):

        # Stop streaming
        if not self.conn.stop():
            del self.conn # try to force the streaming to end by deleting the ecube connection object
            self.conn = eCubeStream(debug=True)

        # Remove the added sources
        self.conn.remove(('Headstages', self.headstage))
        
    def get(self):
        '''data
        Retrieve a packet from the server
        '''
        try:
            return next(self.gen)
        except StopIteration:
            data_block = self.conn.get() # in the form of (time_stamp, data_source, data_content)
            # while data_block[1] != "Headstages":
            #     data_block = self.conn.get() # ideally this shouldn't happen.
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

class Digital(Broadband):
    '''
    Wrapper class for pyecubestream compatible with using in DataSource for
    buffering digital data. Compatible with riglib.source.MultiChanDataSource
    '''
    # Required by DataSourceSystem: update_freq and dtype (see make() below)
    update_freq = 25000.
    dtype = np.dtype('bool')

    def __init__(self, channels=[1]):
        '''
        Inputs:
            channels [int array]: channel list (1-indexed)
        '''
        # Initialize the servernode-control connection
        self.conn = eCubeStream(debug=False)
        self.channels = channels

    def start(self):
        # Remove all existing sources
        subscribed = self.conn.listadded()
        if len(subscribed[0]) > 0:
            self.conn.remove(('Headstages', self.headstage))
        if len(subscribed[1]) > 0:
            self.conn.remove(('AnalogPanel',))
        if len(subscribed[2]) > 0:
            self.conn.remove(('DigitalPanel',))

        # Add the requested headstage channels if they are available
        available = self.conn.listavailable()[2] # (headstages, analog, digital)
        for ch in self.channels:
            if ch > available:
                raise RuntimeError('requested channel {} is not available ({} connected)'.format(
                    ch, available))
            self.conn.add(('DigitalPanelAsChans', (ch, ch))) # add channels one at a time
        subscribed = self.conn.listadded() # in debug mode this prints out the added channels

        # Start streaming
        self.conn.start()

        # Start with an empty generator
        self.gen = iter(())

        # Call get once to make sure we have the right channels
        data_block = self.conn.get()
        assert data_block[1] == "DigitalPanelAsChans"
        assert data_block[2].shape[1] == len(self.channels)
        print(f"Started streaming from ecube, packet size {data_block[2].shape[0]}")
    
    def stop(self):

        # Stop streaming
        if not self.conn.stop():
            del self.conn # try to force the streaming to end by deleting the ecube connection object
            self.conn = eCubeStream(debug=True)

        # Remove the added sources
        self.conn.remove(('DigitalPanelAsChans',))

class Analog(Broadband):
    '''
    Wrapper class for pyecubestream compatible with using in DataSource for
    buffering analog data. Compatible with riglib.source.MultiChanDataSource
    '''

    def __init__(self, channels=[1]):
        '''
        Inputs:
            channels [int array]: channel list (1-indexed)
        '''
        # Initialize the servernode-control connection
        self.conn = eCubeStream(debug=False)
        self.channels = channels

    def start(self):
        # Remove all existing sources
        subscribed = self.conn.listadded()
        if len(subscribed[0]) > 0:
            self.conn.remove(('Headstages', self.headstage))
        if len(subscribed[1]) > 0:
            self.conn.remove(('AnalogPanel',))
        if len(subscribed[2]) > 0:
            self.conn.remove(('DigitalPanel',))

        # Add the requested headstage channels if they are available
        available = self.conn.listavailable()[1] # (headstages, analog, digital)
        for ch in self.channels:
            if ch > available:
                raise RuntimeError('requested channel {} is not available ({} connected)'.format(
                    ch, available))
            self.conn.add(('AnalogPanel', (ch, ch))) # add channels one at a time
        subscribed = self.conn.listadded() # in debug mode this prints out the added channels

        # Start streaming
        self.conn.start()

        # Start with an empty generator
        self.gen = iter(())

        # Call get once to make sure we have the right channels
        data_block = self.conn.get()
        assert data_block[1] == "AnalogPanel"
        assert data_block[2].shape[1] == len(self.channels)
        print(f"Started streaming from ecube, packet size {data_block[2].shape[0]}")
    
    def stop(self):

        # Stop streaming
        if not self.conn.stop():
            del self.conn # try to force the streaming to end by deleting the ecube connection object
            self.conn = eCubeStream(debug=True)

        # Remove the added sources
        self.conn.remove(('AnalogPanel',))

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

class LFP_Plus_Trigger(DataSourceSystem):
    '''
    Adds a single analog trigger as channel 0. Compatible with riglib.source.MultiChanDataSource
    '''
    # Required by DataSourceSystem: update_freq and dtype (see make() below)
    update_freq = 1000.
    dtype = np.dtype('float')

    def __init__(self, headstage=7, trigger_ach=0, channels=[1]):
        '''
        Constructor for ecube.Broadband
        Inputs:
            headstages int: headstage number (1-indexed)
            channels [int array]: channel list (1-indexed)
        '''
        # Initialize the servernode-control connection
        self.conn = eCubeStream(debug=False)
        self.headstage = headstage
        self.trig_channel = trigger_ach
        self.channels = channels

    def start(self):
        print("Starting ecube streaming datasource...")

        # Remove all existing sources
        subscribed = self.conn.listadded()
        if len(subscribed[0]) > 0:
            self.conn.remove(('Headstages', self.headstage))
        if len(subscribed[1]) > 0:
            self.conn.remove(('AnalogPanel',))
        if len(subscribed[2]) > 0:
            self.conn.remove(('DigitalPanel',))

        # Add the requested headstage channels if they are available
        available = self.conn.listavailable()[0][self.headstage-1] # (headstages, analog, digital); hs are 1-indexed
        for ch in self.channels:
            if ch > available:
                raise RuntimeError('requested channel {} is not available ({} connected)'.format(
                    ch, available))
            self.conn.add(('Headstages', self.headstage, (ch, ch))) # add channels one at a time

        # Add the digital panel for triggering
        self.conn.add(('AnalogPanel', (self.trig_channel, self.trig_channel)))
        subscribed = self.conn.listadded() # in debug mode this prints out the added channels

        # Start streaming
        self.conn.start()

        # Start with an empty generator for the headstage channels
        self.gen = iter(())

    
    def stop(self):

        # Stop streaming
        if not self.conn.stop():
            del self.conn # try to force the streaming to end by deleting the ecube connection object
            self.conn = eCubeStream(debug=True)

        # Remove the added sources
        self.conn.remove(('Headstages', self.headstage))
        self.conn.remove(('AnalogPanel',))
        
    def get(self):
        '''data
        Retrieve a packet from the server
        '''
        try:
            return next(self.gen)
        except StopIteration:
            data_block = self.conn.get() # in the form of (time_stamp, data_source, data_content)
            if data_block[1] == "Headstages":
                self.gen = multi_chan_generator(data_block[2], self.channels, downsample=25)
                return next(self.gen)
            else:
                return 0, data_block[2][::25] # The trigger data
    
    

class LFP_Plus_Trigger_File(DataSourceSystem):
    '''
    Adds a single analog trigger as channel 0, but reading from file. Compatible with riglib.source.MultiChanDataSource
    '''
    # Required by DataSourceSystem: update_freq and dtype (see make() below)
    update_freq = 1000.
    chunksize = 728
    dtype = np.dtype('float')

    def __init__(self, channels, ecube_bmi_filename=None, trig_channel=0):
        
        if not ecube_bmi_filename:
            ecube_bmi_filename = "/data/raw/ecube/2022-08-19_BMI3D_te6569"

        # Open the file
        self.trig_channel = trig_channel
        self.channels = channels
        zero_idx_channels = [ch-1 for ch in self.channels]
        self.file = aopy.data.load_ecube_data_chunked(ecube_bmi_filename, "Headstages", channels=zero_idx_channels, chunksize=self.chunksize)
        self.trig_file = aopy.data.load_ecube_data_chunked(ecube_bmi_filename, "AnalogPanel", channels=[self.trig_channel], chunksize=self.chunksize)
        self.trig_flag = True
        
    def get(self):
        '''data
        Read a "packet" worth of data from the file
        '''
        try:
            return next(self.gen)
        except (StopIteration, AttributeError):
            time.sleep(1./(25000/self.chunksize))
            if self.trig_flag:
                try:
                    trig_chunk = next(self.trig_file)[::25]
                except StopIteration:
                    trig_chunk = np.zeros((int(728/25),1))
                self.trig_flag = False
                return 0, np.squeeze(trig_chunk)
            try:
                data_block = next(self.file)
                self.trig_flag = True
            except StopIteration:
                data_block = np.zeros((int(728/25),len(self.channels)))
            self.gen = multi_chan_generator(data_block, self.channels, downsample=25)
            return next(self.gen)
    
class LFP_Blanking(LFP_Plus_Trigger):
    '''
    Blanks LFP data in the interval when the trigger is on. Compatible with riglib.source.MultiChanDataSource
    '''

    blanking = False
    analog_buffer = np.nan*np.zeros((1000*30,))
    headstage_buffer = []
        
    def get(self):
        '''data
        Retrieve a packet from the server
        '''
        try:
            if self.blanking:
                chan, data = next(self.gen)
                blank = self.headstage_buffer[:,chan-1] # copy the last good headstage data for this channel
                return (chan, blank)
            else:
                return next(self.gen)        
        except StopIteration:
            data_block = self.conn.get() # in the form of (time_stamp, data_source, data_content)
            while data_block[1] != "Headstages": # new packet of trigger data
                trig = data_block[2][::25]
                n_trig = len(trig)
                self.analog_buffer[:-n_trig] = self.analog_buffer[n_trig:]
                self.analog_buffer[-n_trig:] = trig
                median = np.nanmedian(self.analog_buffer)
                std = np.nanstd(self.analog_buffer)
                thr = median + 3*std
                if np.max(trig) > thr:
                    self.blanking = True
                else:
                    self.blanking = False
                data_block = self.conn.get() # in the form of (time_stamp, data_source, data_content)

            self.gen = multi_chan_generator(data_block[2], self.channels, downsample=25)
            if self.blanking:
                chan, data = next(self.gen)
                blank = self.headstage_buffer[:,chan-1] # copy the last good headstage data for this channel
                return (chan, blank)
            else:
                self.headstage_buffer = data_block[2][::25,:] # save this good hs data for later if blanking is needed
                return next(self.gen)


class LFP_Blanking_File(LFP_Plus_Trigger_File):
    '''
    Blanks LFP data in the interval when the trigger is on. Compatible with riglib.source.MultiChanDataSource
    '''

    blanking = False
    analog_buffer = np.nan*np.zeros((1000*30,))
    headstage_buffer = []

    def get(self):
        '''data
        Read a "packet" worth of data from the file
        '''
        try:
            if self.blanking:
                chan, data = next(self.gen)
                blank = self.headstage_buffer[:,chan-1] # copy the last good headstage data for this channel
                return (chan, blank)
            else:
                return next(self.gen)        
        except (StopIteration, AttributeError):
            time.sleep(1./(25000/self.chunksize))
            if self.trig_flag:
                try:
                    trig_chunk = next(self.trig_file)[::25]
                except StopIteration:
                    trig_chunk = np.zeros((int(728/25),1))
                self.trig_flag = False
                
                n_trig = trig_chunk.size
                self.analog_buffer[:-n_trig] = self.analog_buffer[n_trig:]
                self.analog_buffer[-n_trig:] = np.squeeze(trig_chunk)
                median = np.nanmedian(self.analog_buffer)
                std = np.nanstd(self.analog_buffer)
                thr = median + 3*std
                if np.max(trig_chunk) > thr:
                    self.blanking = True
                else:
                    self.blanking = False

            try:
                data_block = next(self.file)
                self.trig_flag = True
            except StopIteration:
                data_block = np.zeros((int(728/25),len(self.channels)))
            self.gen = multi_chan_generator(data_block, self.channels, downsample=25)
            if self.blanking:
                chan, data = next(self.gen)
                blank = self.headstage_buffer[:,chan-1] # copy the last good headstage data for this channel
                return (chan, blank)
            else:
                self.headstage_buffer = data_block[2][::25,:] # save this good hs data for later if blanking is needed
                return next(self.gen)
    
def make_source_class(cls, trigger_ach):
    
    def init(self, **kwargs):
        super(self.__class__, self).__init__(trigger_ach=trigger_ach, **kwargs)
        
    return type(cls.__name__, (cls,), dict(__init__=init))

# e.g.
# cls = make_source_class(LFP_Plus_Trigger, 17) # laser ch 2