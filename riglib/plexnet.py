import struct
import socket
from collections import namedtuple

import numpy as np

PACKETSIZE = 512

WaveData = namedtuple("WaveData", ["type", "ts", "chan", "unit","waveform"])

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
    
    dbnames = 'type,Uts,ts,chan,unit,nwave,nword'.split(',')
    dbtypes = 'hHI4h'

    def __init__(self, addr, port):
        self.addr = (addr, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.addr)
        self.streaming = False
    
    def _recv(self):
        d = ''
        while len(d) < PACKETSIZE:
            d += self.sock.recv(PACKETSIZE - len(d))
        return d
    
    def init(self, channels, waveforms=True, analog=False):
        packet = np.zeros(PACKETSIZE / 4., dtype=np.int32)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_CONNECT_CLIENT
        packet[1] = True #timestamp
        packet[2] = waveforms
        packet[3] = analog
        packet[4] = 1 #channels start
        packet[5] = channels+1
        print "Sent transfer mode command... "
        self.sock.sendall(packet.tostring())
        resp = struct.unpack('%di'%(PACKETSIZE/4), self._recv())
        print "recieved %s"%repr(resp)

        packet[:] = 0
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_GET_PARAMETERS_MMF
        self.sock.sendall(packet.tostring())

        print "Request parameters..."
        gotServerArea = False
        while not gotServerArea:
            resp = struct.unpack('128i', self._recv())
            
            if resp[0] == self.PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_SENDING_SERVER_AREA:
                self.n_spike = resp[15]
                self.n_cont = resp[17]
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

        self.sock.sendall(raw)
    
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

        self.sock.sendall(raw)

    def start(self):
        packet = np.zeros(PACKETSIZE, dtype=np.int32)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_START_DATA_PUMP
        self.sock.sendall(packet.tostring())
        self.streaming = True
        print "Started plexon stream"

    def stop(self):
        packet = np.zeros(PACKETSIZE, dtype=np.int32)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_STOP_DATA_PUMP
        self.sock.sendall(packet.tostring())
        self.streaming = False
        print "Stopped plexon stream"

    def disconnect(self):
        packet = np.zeros(PACKETSIZE, dtype=np.int32)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_DISCONNECT_CLIENT
        self.sock.sendall(packet.tostring())
        self.sock.close()
        print "Disconnected from plexon"
    
    def __del__(self):
        self.disconnect()

    def get_data(self):
        while self.streaming:
            buf = self._recv()
            #ibuf = np.fromstring(buf[:16], dtype=np.int32)
            ibuf = struct.unpack('4i', buf[:16])
            if ibuf[0] != 1:
                yield None
            
            num_server_dropped = ibuf[2]
            num_mmf_dropped = ibuf[3]
            #print "new packet %r"%ibuf
            buf = buf[16:]
            
            while len(buf) > 16:
                #header = np.fromstring(buf[:16], dtype=self.dbtype)
                header = dict(zip(self.dbnames, struct.unpack(self.dbtypes, buf[:16])))
                buf = buf[16:]
                
                if header['type'] in [0, -1]:
                    #empty block
                    yield None
                    break;
                
                wavedat = None
                if header['nwave'] > 0:
                    l = header['nwave'] * header['nword'] * 2
                    wavedat = struct.unpack('%dh'%(l/2.), buf[:l])
                buf = buf[l:]
                
                ts = long(header['Uts']) << 32 | header['ts']
                
                yield WaveData(type=header['type'], chan=header['chan'],
                    unit=header['unit'], ts=ts, waveform = wavedat)
    
class System(object):
    def __init__(self, addr=("10.0.0.2", 6000), channels=256, waveforms=False, analog=False):
        self.conn = Connection(*addr)
        self.conn.init(channels, waveforms, analog)
        self.set_spikes(waveforms=waveforms)
        self.set_continuous()

if __name__ == "__main__":
    import itertools
    conn = Connection("10.0.0.13", 6000)
    conn.init(256)
    conn.set_spikes()
    conn.start()
    it = conn.get_data()
    for i, wave in itertools.izip(xrange(100000), it):
        wave
    conn.stop()
    conn.disconnect()
    
    
