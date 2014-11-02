'''
RDA (Remote Data Access) client code to receive data from the Brain Products
BrainVision Recorder.
'''

import time
import numpy as np
from struct import *
from socket import *


# Helper function for receiving whole message
def RecvData(socket, requestedSize):
    returnStream = ''
    while len(returnStream) < requestedSize:
        databytes = socket.recv(requestedSize - len(returnStream))
        if databytes == '':
            raise RuntimeError, "connection broken"
        returnStream += databytes
 
    return returnStream   

    
# Helper function for splitting a raw array of
# zero terminated strings (C) into an array of python strings
def SplitString(raw):
    stringlist = []
    s = ""
    for i in range(len(raw)):
        if raw[i] != '\x00':
            s = s + raw[i]
        else:
            stringlist.append(s)
            s = ""

    return stringlist
    

# Helper function for extracting eeg properties from a raw data array
# read from tcpip socket
def GetProperties(rawdata):

    # Extract numerical data
    (channelCount, samplingInterval) = unpack('<Ld', rawdata[:12])

    # Extract resolutions
    resolutions = []
    for c in range(channelCount):
        index = 12 + c * 8
        restuple = unpack('<d', rawdata[index:index+8])
        resolutions.append(restuple[0])

    # Extract channel names
    channelNames = SplitString(rawdata[12 + 8 * channelCount:])

    return (channelCount, samplingInterval, resolutions, channelNames)



class EMGData(object):
    '''For use with a MultiChanDataSource in order to acquire streaming EMG/EEG/EOG
    data (not limited to just EEG) from the BrainProducts BrainVision Recorder.
    '''

    update_freq = 2500.  # TODO -- check
    #dtype = np.dtype('float')
    dtype = np.dtype([('data',       sub_dtype_data),
                      ('ts_arrival', np.float64)])

    RDA_MessageStart     = 1      # 
    RDA_MessageData      = 2      # message type for 16-bit data
    RDA_MessageStop      = 3      # 
    RDA_MessageData32    = 4      # message type for 32-bit data
    RDA_MessageKeepAlive = 10000  # packets of this message type can discarded


    # TODO -- added **kwargs argument to __init__ for now because MCDS is passing
    #   in source_kwargs which contains 'channels' kwarg which is not needed/expected
    #   need to fix this later
    def __init__(self, recorder_ip='192.168.137.1', nbits=16, **kwargs):
        self.recorder_ip = recorder_ip

        if nbits == 16:
            self.port = 51234
            self.fmt = '<h'  # little-endian byte order, signed 16-bit integer
            self.step = 2    # bytes
        elif nbits == 32:
            self.port = 51244
            self.fmt = '<f'  # little-endian byte order, 32-bit IEEE float
            self.step = 4    # bytes
        else:
            raise Exception('Invalid value for nbits -- must be either 16 or 32!')

        self.nbits = nbits

        # Create a tcpip socket
        self.sock = socket(AF_INET, SOCK_STREAM)
        
        # This needs to happen in get_data so that we don't miss the first
        # packet with msgtype == RDA_MessageStart
        # Connect to the Recorder host
        # self.sock.connect((self.recorder_ip, self.port))
    
    def start(self):
        '''Start the buffering of data.'''

        self.streaming = True
        self.data = self.get_data()


    def stop(self):
        '''Stop the buffering of data.'''

        self.streaming = False

    def disconnect(self):
        '''Close the connection to Recorder.'''
        
        if not self._init:
            raise ValueError("Please connect to Recorder first.")
        
        # self._init = False
    
    def __del__(self):
        self.disconnect()

    # TODO -- add comment about how this will get called by the source
    def get(self):
        return self.data.next()

    def get_data(self):
        '''A generator which yields packets as they are received'''
        
        assert self._init, "Please initialize the connection first"
        
        self.sock.connect((self.recorder_ip, self.port))

        chan_idx = 0

        while self.streaming:

            # Get message header as raw array of chars
            rawhdr = RecvData(self.sock, 24)

            # Split array into useful information; id1 to id4 are constants
            (id1, id2, id3, id4, msgsize, msgtype) = unpack('<llllLL', rawhdr)
            
            # Get data part of message, which is of variable size
            rawdata = RecvData(self.sock, msgsize - 24)
            arrival_ts = time.time()

            # Perform action dependend on the message type
            if msgtype == self.RDA_MessageStart:
                # Start message, extract eeg properties and display them
                (channelCount, samplingInterval, resolutions, channelNames) = GetProperties(rawdata)
                
                # reset block counter
                lastBlock = -1

                print "Start"
                print "Number of channels: " + str(channelCount)
                print "Sampling interval: " + str(samplingInterval)
                print "Resolutions: " + str(resolutions)
                print "Channel Names: " + str(channelNames)

                channels = [int(name) for name in channelNames]

            elif msgtype == self.RDA_MessageStop:
                
                # TODO -- what to do here? set self.streaming = False?
                # call self.stop_data()? self.disconnect()?
                pass

            elif (msgtype == self.RDA_MessageData) or (msgtype == self.RDA_MessageData32):

                # Extract numerical data
                (block, points, markerCount) = unpack('<LLL', rawdata[:12])

                # Extract eeg data
                for i in range(points * channelCount):
                    index = 12 + (self.step * i)
                    AD_value = unpack(self.fmt, rawdata[index:index+self.step])[0]
                    chan = channels[chan_idx]
                    uV_value = AD_value * resolutions[chan_idx]
                    
                    # yield EMGData(chan=chan, uV_value=uV_value, arrival_ts=arrival_ts)
                    # TODO -- check
                    # yield (chan, np.array([uV_value], dtype='float'))
                    yield np.array([(data,
                                     ts_arrival)],
                                   dtype=self.dtype)

                    chan_idx = (chan_idx + 1) % channelCount


                # disregard marker data for now

                # # Check for overflow
                # if lastBlock != -1 and block > lastBlock + 1:
                #     print "*** Overflow with " + str(block - lastBlock) + " datablocks ***" 
                # lastBlock = block

            elif msgtype == self.RDA_MessageKeepAlive:
                pass
            
            else:
                raise Exception('Unrecognized RDA message type: ' + str(msgtype))
