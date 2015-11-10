'''
RDA (Remote Data Access) client code to receive data from the Brain Products
BrainVision Recorder.
'''

import time
import numpy as np
from struct import *
from socket import *
import array
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
#import math
from riglib.filter import Filter
from ismore import settings, brainamp_channel_lists


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
    dtype = np.dtype([('data',       np.float64),
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
        self.nbits = nbits

        if self.nbits == 16:
            self.port = 51234
            self.fmt = '<h'  # little-endian byte order, signed 16-bit integer
            self.step = 2    # bytes
        elif self.nbits == 32:
            self.port = 51244
            self.fmt = '<f'  # little-endian byte order, 32-bit IEEE float
            self.step = 4    # bytes
        else:
            raise Exception('Invalid value for nbits -- must be either 16 or 32!')

        # Create a tcpip socket
        self.sock = socket(AF_INET, SOCK_STREAM)

        self.fs = 1000
        # calculate coefficients for a 4th-order Butterworth BPF from 10-450 Hz
        band  = [10, 450]  # Hz
        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        self.bpf_coeffs = butter(4, [low, high], btype='band')
        # self.band_pass_filter = Filter(self.bpf_coeffs[0], self.bpf_coeffs[1])

        # calculate coefficients for multiple 2nd-order notch filers
        self.notchf_coeffs = []
        for freq in [50, 150, 250, 350]:
            band  = [freq - 1, freq + 1]  # Hz
            nyq   = 0.5 * self.fs
            low   = band[0] / nyq
            high  = band[1] / nyq
            self.notchf_coeffs.append(butter(2, [low, high], btype='bandstop'))

        if 'brainamp_channels' in kwargs:
            self.channels = kwargs['brainamp_channels']
        
    def start(self):
        '''Start the buffering of data.'''

        self.streaming = True
        self.data = self.get_data()

    def stop(self):
        '''Stop the buffering of data.'''

        self.streaming = False

    def disconnect(self):
        '''Close the connection to Recorder.'''
    
    def __del__(self):
        self.disconnect()

    # TODO -- add comment about how this will get called by the source
    def get(self):
        return self.data.next()

    def get_data(self):
        '''A generator which yields packets as they are received'''
        
        self.sock.connect((self.recorder_ip, self.port))


        chan_idx = 0

        

        self.notch_filters = []
        for b, a in self.notchf_coeffs:
            self.notch_filters.append(Filter(b=b, a=a))

        channelCount = len(settings.BRAINAMP_CHANNELS)
        self.channel_filterbank_emg = [None]*channelCount
        for k in range(channelCount):
            filts = [Filter(self.bpf_coeffs[0], self.bpf_coeffs[1])]
            for b, a in self.notchf_coeffs:
                filts.append(Filter(b=b, a=a))
            self.channel_filterbank_emg[k] = filts

        self.channel_filterbank_eeg = [None]*channelCount
        for k in range(channelCount):
            filts = [Filter(self.bpf_coeffs[0], self.bpf_coeffs[1])]
            filts.append(Filter(b=self.notchf_coeffs[0], a=self.notchf_coeffs[1]))
            self.channel_filterbank_eeg[k] = filts

        while self.streaming:

            # Get message header as raw array of chars
            rawhdr = RecvData(self.sock, 24)

            # Split array into useful information; id1 to id4 are constants
            (id1, id2, id3, id4, msgsize, msgtype) = unpack('<llllLL', rawhdr)
            
            # Get data part of message, which is of variable size
            rawdata = RecvData(self.sock, msgsize - 24) 
            
            ts_arrival = time.time()
       

            # Perform action dependend on the message type
            if msgtype == self.RDA_MessageStart:
                # Start message, extract eeg properties and display them
                (channelCount, samplingInterval, resolutions, channelNames) = GetProperties(rawdata)
                
                # if channelNames != self.channels: #settings.BRAINAMP_CHANNELS
                #     print 'ERROR: Selected channels do not match the streamed channel names. Double check!'

                # else:
                #     #channelNames_filt = list()
                #     for i in range(channelCount):
                #         channelNames.append(channelNames[i] + "_filt")
                #     channelCount_filt = channelCount*2 
                #     resolutions_filt = [resolutions[0]]*(len(channelNames))
                #     resolutions = resolutions + resolutions_filt

                channels = self.channels
                
                channelCount_filt = channelCount*2 
                resolutions_filt = [resolutions[0]]*(len(channelNames))
                resolutions = resolutions + resolutions_filt   
                
                # reset block counter
                lastBlock = -1
                
                
                # channels_filt_all = list()
                # for i in range(len(channelNames)):
                #     channels_filt = [channelNames[i] + "_filt" ]
                #     channels_filt_all.append(channels_filt)
                    
                
                print "Start"
                print "Number of channels: " + str(channelCount_filt)
                print "Sampling interval: " + str(samplingInterval)
                print "Resolutions: " + str(resolutions)
                print "Channel Names: " + str(channelNames)
                print "Sampling Frequency: " + str(1000000/samplingInterval)

                #channels = [int(name) for name in channelNames]
                #channels = channelNames

                

                #print type(channels)
                
                #channels = [int(name) for name in channels_filt_all]#andrea

                #print type(channels)


            elif msgtype == self.RDA_MessageStop:
                
                # TODO -- what to do here? set self.streaming = False?
                # call self.stop_data()? self.disconnect()?
                pass

            elif (msgtype == self.RDA_MessageData) or (msgtype == self.RDA_MessageData32):

                # Extract numerical data
                (block, points, markerCount) = unpack('<LLL', rawdata[:12])

                # Extract eeg/emg data

                # OLD, INEFFICIENT METHOD (yielding data points one at a time)
                # for i in range(points * channelCount):
                #     index = 12 + (self.step * i)
                #     AD_value = unpack(self.fmt, rawdata[index:index+self.step])[0]
                #     chan = channels[chan_idx]
                #     uV_value = AD_value * resolutions[chan_idx]
                #     yield (chan, np.array([(uV_value, ts_arrival)], dtype=self.dtype))
                #     chan_idx = (chan_idx + 1) % channelCount



                

                # MORE EFFICIENT -- yield all data points for a channel at once
                # data_ = array.array('h')
                # data_.fromstring(rawdata[12:])  # TODO -- make more general
                # data = np.zeros((channelCount, points), dtype=self.dtype)
                # data['data'] = np.array(data_).reshape((points, channelCount)).T
                # data['ts_arrival'] = ts_arrival
                # for chan_idx in range(channelCount):
                #     data[chan_idx, :]['data'] *= resolutions[chan_idx]
                #     chan = channels[chan_idx]
                #     yield (chan, data[chan_idx, :])

                # Filter the data as the packages arrive - andrea
                data_ = array.array('h')
                data_.fromstring(rawdata[12:])  # TODO -- make more general
                data = np.zeros((channelCount, points), dtype=self.dtype)
                data['ts_arrival'] = ts_arrival
                data['data'] = np.array(data_).reshape((points, channelCount)).T
                
                
                datafilt = np.zeros((channelCount_filt, points), dtype=self.dtype)
                datafilt['ts_arrival'] = ts_arrival
                filtered_data = np.zeros((channelCount, points), dtype=self.dtype)
                filtered_data['data'] = data['data'].copy()


                for k in range(channelCount):
                    if  channels[k] in brainamp_channel_lists.emg14_raw: #Apply filters for emg
                        for filt in self.channel_filterbank_emg[k]:
                            filtered_data[k]['data'] =  filt(filtered_data[k]['data'] )
                                
                    else:
                        pass #apply filters for eeg. To be implemented      
                
                datafilt['data'] = np.vstack([data['data'], filtered_data['data']])
                    
                    # from scipy.io import savemat
                    # import os
                    # savemat(os.path.expandvars('$HOME/code/ismore/emg_rda_filt.mat'), dict(emg_filt = filtered_data, datafilt = datafilt['data'], filterbank = self.channel_filterbank))
                    
                for chan_idx in range(channelCount_filt):
                    datafilt[chan_idx, :]['data'] *= resolutions[chan_idx]
                    chan = channels[chan_idx]
                    yield (chan, datafilt[chan_idx, :])
             

                # disregard marker data for now

                # # Check for overflow
                # if lastBlock != -1 and block > lastBlock + 1:
                #     print "*** Overflow with " + str(block - lastBlock) + " datablocks ***" 
                # lastBlock = block

            elif msgtype == self.RDA_MessageKeepAlive:
                pass
            
            else:
                raise Exception('Unrecognized RDA message type: ' + str(msgtype))
