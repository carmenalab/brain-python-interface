import struct
import socket
from collections import namedtuple

import numpy as np

PACKETSIZE = 512

class Connection(object):
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
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect(self.addr)
    
    def init(self, channels, waveforms=True, analog=True):
        packet = np.zeros(PACKETSIZE / 4., dtype=np.int32)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_CONNECT_CLIENT
        packet[1] = True #timestamp
        packet[2] = waveforms
        packet[3] = analog
        packet[4] = 0 #channels start
        packet[5] = channels+1
        print "Sent transfer mode command..."
        self.sock.sendto(packet.tostring(), self.addr)
        resp = np.fromstring(self.sock.recvfrom(PACKETSIZE)[0], dtype=np.int32)
        print "recieved %r"%resp

        packet[:] = 0
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_GET_PARAMETERS_MMF
        self.socket.sendto(packet.tostring(), self.addr)

        print "Request parameters..."
        gotServerArea = False
        while not gotServerArea:
            resp = np.fromstring(self.sock.recvfrom(PACKETSIZE), dtype=np.int32)
            if resp[0] == self.PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_SENDING_SERVER_AREA:
                self.n_spike, self.n_cont = resp[[15,17]]
                gotServerArea = True
        print "Done init!"
        
    def set_spikes(self, channels=None, waveforms=True):
        packet = np.zeros(5, dtype=np.int32)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_SPIKE_CHANNELS
        packet[2] = 1
        packet[3] = self.n_spike
        raw = packet.tostring()

        packet = np.zeros(PACKETSIZE - len(raw), dtype=np.uint8)
        #always send timestamps, waveforms are optional
        bitmask = 1 | waveforms << 1
        if channels is None:
            packet[:] = bitmask
        else:
            packet[channels] = bitmask
        raw += packet.tostring()

        self.sock.sendto(raw, self.addr)
    
    def set_continuous(self, channels=None):
        packet = np.zeros(5, dtype=np.int32)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_CONTINUOUS_CHANNELS
        packet[2] = 1
        packet[3] = self.n_cont
        raw = packet.tostring()

        packet = np.zeros(PACKETSIZE - len(raw), dtype=np.uint8)
        if channels is None:
            packet[:] = 1
        else:
            packet[channels] = 1
        raw += packet.tostring()

        self.sock.sendto(raw, self.addr)

    def start(self):
        packet = np.zeros(PACKETSIZE, dtype=np.int32)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_START_DATA_PUMP
        self.sock.sendto(packet.tostring(), self.addr)
        print "Started plexon stream"

    def stop(self):
        packet = np.zeros(PACKETSIZE, dtype=np.int32)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_STOP_DATA_PUMP
        self.sock.sendto(packet.tostring(), self.addr)
        print "Stopped plexon stream"

    def disconnect(self):
        packet = np.zeros(PACKETSIZE, dtype=np.int32)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_DISCONNECT_CLIENT
        self.sock.sendto(packet.tostring(), self.addr)
        print "Disconnected from plexon"

    def get_data(self):
        buf = self.sock.recvfrom(PACKETSIZE)
        ibuf = np.fromstring(buf, dtype=np.int32)
        if ibuf[0] != 1:
            return None

        

    
class System(object):
    def __init__(self, addr=("10.0.0.2", 6000), channels=256, waveforms=False, analog=False):
        self.conn = Connection(*addr)
        self.conn.init(channels, waveforms, analog)
        self.set_spikes(waveforms=waveforms)
        self.set_continuous()