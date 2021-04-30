'''
Client-side code to configure and receive neural data from the Plexon PC over
the network.
'''

import re
import math
import array
import struct
import socket
import time
from collections import namedtuple
import os
import numpy as np
import matplotlib.pyplot as plt

PACKETSIZE = 512



WaveData = namedtuple("WaveData", ["type", "ts", "chan", "unit", "waveform", "arrival_ts"])
chan_names = re.compile(r'^(\w{2,4})(\d{2,3})(\w)?')

class Connection(object):
    '''
    A wrapper around a UDP socket which sends the Omniplex PC commands and 
    receives data. Must run in a separte process (e.g., through `riglib.source`) 
    if you want to use it as part of a task (e.g., BMI control)
    '''
    PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_CONNECT_CLIENT = (10000)
    PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_DISCONNECT_CLIENT = (10999)
    PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_GET_PARAMETERS_MMF = (10100)
    PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_START_DATA_PUMP = (10200)
    PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_STOP_DATA_PUMP = (10300)
    PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_SPIKE_CHANNELS = (10400)
    PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_CONTINUOUS_CHANNELS = (10401)

    PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_MMF_SIZES = (10001)
    PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_SENDING_SERVER_AREA = (20003)
    PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_SENDING_DATA = (1)

    SPIKE_CHAN_SORTED_TIMESTAMPS = (0x01)
    SPIKE_CHAN_SORTED_WAVEFORMS = (0x02)
    SPIKE_CHAN_UNSORTED_TIMESTAMPS = (0x04)
    SPIKE_CHAN_UNSORTED_WAVEFORMS = (0x08)

    def __init__(self, addr, port):
        self.addr = (addr, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.addr)

        self.num_server_dropped = 0
        self.num_mmf_dropped = 0

        self._init = False
    
    def _recv(self):
        '''
        Receives a single PACKETSIZE chunk from the socket
        '''
        d = ''
        while len(d) < PACKETSIZE:
            d += self.sock.recv(PACKETSIZE - len(d))
        return d
    
    def connect(self, channels, waveforms=False, analog=True):
        '''Establish a connection with the plexnet remote server, then request and set parameters

        Parameters
        ----------
        channels : int
            Number of channels to initialize through the server
        waveforms : bool, optional
            Set to true if you want to stream spike waveforms (not available for MAP system?)
        analog : bool, optional
            Set to true if you want to receive data from the analog channels

        Returns
        -------
        None            
        '''

        packet = array.array('i', '\x00'*PACKETSIZE)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_CONNECT_CLIENT
        packet[1] = True #timestamp
        packet[2] = waveforms
        packet[3] = analog
        packet[4] = 1 #channels start
        packet[5] = channels+1

        self.sock.sendall(packet.tostring())
        
        resp = array.array('i', self._recv())

        if resp[0] == self.PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_MMF_SIZES:
            self.n_cmd = resp[3]
            if 0 < self.n_cmd < 32:
                sup_spike = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_SPIKE_CHANNELS
                sup_cont = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_CONTINUOUS_CHANNELS
                self.supports_spikes = any([b == sup_spike for b in resp[4:]])
                self.supports_cont = any([b == sup_cont for b in resp[4:]])

        print('supports spikes:', self.supports_spikes)
        print('supports continuous:', self.supports_cont)

        packet = array.array('i', '\x00'*PACKETSIZE)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_GET_PARAMETERS_MMF
        self.sock.sendall(packet.tostring())

        self.params = []

        gotServerArea = False
        while not gotServerArea:
            resp = array.array('i', self._recv())
            self.params.append(resp)
            
            if resp[0] == self.PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_SENDING_SERVER_AREA:
                self.n_spike = resp[15]
                self.n_cont = resp[17]
                print("Spike channels: %d, continuous channels: %d"%(self.n_spike, self.n_cont))
                gotServerArea = True
        
        self._init = True

        
    def select_spikes(self, channels=None, waveforms=True, unsorted=False):
        '''
        Sets the channels from which to receive spikes. This function always requests sorted data

        Parameters
        ----------
        channels : array_like, optional
            A list of channels which you want to see spikes from
        waveforms : bool, optional
            Request spikes from all selected channels

        Returns
        -------
        None
        '''
        if not self._init:
            raise ValueError("Please initialize the connection first")
        if not self.supports_spikes:
            raise ValueError("Server does not support spike streaming!")
        packet = array.array('i', '\x00'*20)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_SPIKE_CHANNELS
        packet[2] = 1
        packet[3] = self.n_spike
        raw = packet.tostring()

        #always send timestamps, waveforms are optional
        bitmask = 1 | waveforms << 1
        if unsorted:
            bitmask |= 1<<2 | waveforms << 3

        if channels is None:
            raw += array.array('b', [bitmask]*(PACKETSIZE - 20)).tostring()
        else:
            packet = array.array('b', '\x00'*(PACKETSIZE - 20))
            for c in channels:
                packet[c-1] = bitmask
            raw += packet.tostring()

        self.sock.sendall(raw)

    def select_continuous(self, channels=None):
        '''
        Sets the channels from which to receive continuous neural data (e.g., LFP)

        Parameters
        ----------
        channels : array_like, optional
            A list of channels which you want to see spikes from

        Returns
        -------
        None        
        '''
        if not self._init:
            raise ValueError("Please initialize the connection first")
        if not self.supports_cont:
            raise ValueError("Server does not support continuous data streaming!")

        if channels is None:  # select all of them
            # print 'selecting all continuous channels'
            chan_selection = array.array('b', [1]*self.n_cont)
        else:
            # print 'selecting specified continuous channels'
            chan_selection = array.array('b', [0]*self.n_cont)
            for c in channels:
                # always true unless channels outside the range [1,...,self.n_cont] were specified
                if c-1 < len(chan_selection):
                    chan_selection[c-1] = 1

        n_packets = int(math.ceil(float(self.n_cont) / PACKETSIZE))
        HEADERSIZE = 20  # bytes
        chan_offset = 0

        # e.g., for 800 continuous channels, 2 "packets" are formed
        #   chan_offset is 0 for the first packet
        #   chan_offset is 492 for the second packet

        raw = ''
        for packet_num in range(n_packets):
            # print 'subpacket:', packet_num
            header = array.array('i', '\x00'*HEADERSIZE)
            header[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_CONTINUOUS_CHANNELS
            header[1] = packet_num
            header[2] = n_packets
            # header[3] = number of channels that are specified (or not specified) in the selection
            #             bytes that follow
            header[4] = chan_offset # channel offset in this packet

            if chan_offset + (PACKETSIZE-HEADERSIZE) < len(chan_selection[chan_offset:]):
                payload = chan_selection[chan_offset:chan_offset+PACKETSIZE-HEADERSIZE]
                n_selections = len(payload)
                
                chan_offset = chan_offset + len(payload)
            else:  # there are less than PACKETSIZE - HEADERSIZE channels left to specify
                payload = chan_selection[chan_offset:]
                n_selections = len(payload)

                # don't need to worry about incrementing chan_offset (reached end)

                # pad with zeros
                n_pad = PACKETSIZE - HEADERSIZE - len(payload)
                payload += array.array('b', [0]*n_pad)
                

            header[3] = n_selections
            raw += header.tostring()
            raw += payload.tostring()

        self.sock.sendall(raw)

    def start_data(self):
        '''Start the data pump from plexnet remote'''
        if not self._init:
            raise ValueError("Please initialize the connection first")
        packet = array.array('i', '\x00'*PACKETSIZE)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_START_DATA_PUMP
        self.sock.sendall(packet.tostring())
        self.streaming = True

    def stop_data(self):
        '''Stop the data pump from plexnet remote'''
        if not self._init:
            raise ValueError("Please initialize the connection first")
        packet = array.array('i', '\x00'*PACKETSIZE)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_STOP_DATA_PUMP
        self.sock.sendall(packet.tostring())
        self.streaming = False

    def disconnect(self):
        '''Disconnect from the plexnet remote server and close all network sockets'''
        if not self._init:
            raise ValueError("Please initialize the connection first")
        packet = array.array('i', '\x00'*PACKETSIZE)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_DISCONNECT_CLIENT
        self.sock.sendall(packet.tostring())
        self.sock.close()
    
    def __del__(self):
        self.disconnect()

    def get_data(self):
        '''
        A generator which yields packets as they are received
        '''
        
        assert self._init, "Please initialize the connection first"
        hnames = 'type,Uts,ts,chan,unit,nwave,nword'.split(',')
        invalid = set([0, -1])
        
        while self.streaming:
            packet = self._recv()
            
            arrival_ts = time.time()
            ibuf = struct.unpack('4i', packet[:16])
            if ibuf[0] == 1:
                self.num_server_dropped = ibuf[2]
                self.num_mmf_dropped = ibuf[3]
                packet = packet[16:]
                
                while len(packet) > 16:
                    header = dict(list(zip(hnames, struct.unpack('hHI4h', packet[:16]))))
                    packet = packet[16:]
                    
                    if header['type'] not in invalid:
                        wavedat = None
                        if header['nwave'] > 0:
                            l = header['nwave'] * header['nword'] * 2
                            wavedat = array.array('h', packet[:l])
                            packet = packet[l:]

                        chan = header['chan']
                        # when returning continuous data, plexon reports the channel numbers
                        #   as between 0--799 instead of 1--800 (but it doesn't do this
                        #   when returning spike data!), so we have add 1 to the channel number
                        if header['type'] == 5:  # 5 is PL_ADDataType
                            chan = header['chan'] + 1
                        
                        ts = int(header['Uts']) << 32 | int(header['ts'])

                        yield WaveData(type=header['type'], chan=chan,
                            unit=header['unit'], ts=ts, waveform=wavedat, 
                            arrival_ts=arrival_ts)

if __name__ == "__main__":
    import csv
    import time
    import argparse
    parser = argparse.ArgumentParser(description="Collects plexnet data for a set amount of time")
    parser.add_argument("address",help="Server's address")
    parser.add_argument("--port", type=int, help="Server's port (defaults to 6000)", default=6000)
    parser.add_argument("--len", type=float, help="Time (in seconds) to record data", default=30.)
    parser.add_argument("output", help="Output csv file")
    args = parser.parse_args()

    with open(args.output, "w") as f:
        csvfile = csv.DictWriter(f, WaveData._fields)
        csvfile.writeheader()

        #Initialize the connection
        print('initializing connection')
        conn = Connection(args.address, args.port)
        conn.connect(256, analog=True) #Request all 256 channels
        
        print('selecting spike channels')
        spike_channels = [] #2, 3, 4]
        unsorted = False #True
        conn.select_spikes(spike_channels, unsorted=unsorted)
        # conn.select_spikes(unsorted=unsorted)

        print('selecting continuous channels')
        # cont_channels = 512 + np.array([1, 2, 5, 9, 10, 192, 250, 256]) #range(513, 768) #range(512+1, 512+192) #[1, 532, 533, 768, 800] #502, 503, 504, 505] #[85, 86]
        cont_channels = 512 + np.array([53])
        # cont_channels = [1, 532, 533, 768, 800] #502, 503, 504, 505] #[85, 86]
        conn.select_continuous(cont_channels)
        # conn.select_continuous()  # select all 800 continuous channels
 
        # for saving to mat file
        write_to_mat = True 
        n_samp = 2 * 1000*int(args.len)
        n_chan = len(cont_channels)
        data = np.zeros((n_chan, 2*n_samp), dtype='int16')
        idxs = np.zeros(n_chan)
        chan_to_row = dict()
        for i, chan in enumerate(cont_channels):
            chan_to_row[chan] = i


        ts = []
        arrival_ts = []
        t = []
        n_samples = 0
        n_samp = []
        got_first = False


        print('starting data')
        conn.start_data() #start the data pump

        waves = conn.get_data()
        start = time.time()


        while (time.time()-start) < args.len:
            wave = next(waves)
            if not got_first and wave is not None:
                print(wave)
                first_ts = wave.ts
                first_arrival_ts = wave.arrival_ts 
                got_first = True

            if wave is not None:
                csvfile.writerow(dict(wave._asdict()))

            if write_to_mat and wave is not None:
                row = chan_to_row[wave.chan]
                idx = idxs[row]
                n_pts = len(wave.waveform)
                data[row, idx:idx+n_pts] = wave.waveform
                idxs[row] += n_pts

            if wave is not None and wave.chan == 512+53:
                ts.append(wave.ts - first_ts)
                arrival_ts.append(wave.arrival_ts - first_arrival_ts)

                n_samples += len(wave.waveform)
                t.append(time.time() - start)
                n_samp.append(n_samples)


        #Stop the connection
        conn.stop_data()
        conn.disconnect()

        if write_to_mat:
            save_dict = dict()
            save_dict['data'] = data
            save_dict['channels'] = cont_channels

            print('saving data...', end=' ')
            import scipy.io as sio
            sio.matlab.savemat('plexnet_data_0222_8pm_1.mat', save_dict)
            print('done.')


    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(arrival_ts, ts)
    plt.subplot(2,1,2)
    plt.plot(t, n_samp)
    plt.show()
