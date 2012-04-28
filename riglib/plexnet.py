import array
import struct
import socket
import logging
from collections import namedtuple

PACKETSIZE = 512
WaveData = namedtuple("WaveData", ["type", "ts", "chan", "unit","waveform"])
logger = logging.getLogger(__name__)

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
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.addr)
        self.streaming = False

        self.num_server_dropped = 0
        self.num_mmf_dropped = 0

        self._init = False
    
    def _recv(self):
        '''Receives a single PACKETSIZE chunk from the socket'''
        d = ''
        while len(d) < PACKETSIZE:
            d += self.sock.recv(PACKETSIZE - len(d))
        return d
    
    def connect(self, channels, waveforms=True, analog=False):
        '''Establish a connection with the plexnet remote server, then request and set parameters

        Parameters
        ----------
        channels : int
            Number of channels to initialize through the server
        waveforms : bool, optional
            Request spike waveforms?
        analog : bool, optional
            Request analog data?
        '''
        packet = array.array('i', '\x00'*PACKETSIZE)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_CONNECT_CLIENT
        packet[1] = True #timestamp
        packet[2] = waveforms
        packet[3] = analog
        packet[4] = 1 #channels start
        packet[5] = channels+1
        logger.debug("Send transfer mode command")
        self.sock.sendall(packet.tostring())
        
        resp = array.array('i', self._recv())
        logger.debug("Got response from server")
        if resp[0] == self.PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_MMF_SIZES:
            self.n_cmd = resp[3]
            if 0 < self.n_cmd < 32:
                sup_spike = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_SPIKE_CHANNELS
                sup_cont = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_CONTINUOUS_CHANNELS
                self.supports_spikes = any([b == sup_spike for b in resp[4:]])
                self.supports_cont = any([b == sup_cont for b in resp[4:]])

        packet = array.array('i', '\x00'*PACKETSIZE)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_GET_PARAMETERS_MMF
        self.sock.sendall(packet.tostring())

        logger.debug("Request parameters...")
        gotServerArea = False
        while not gotServerArea:
            resp = array.array('i', self._recv())
            
            if resp[0] == self.PLEXNET_COMMAND_FROM_SERVER_TO_CLIENT_SENDING_SERVER_AREA:
                self.n_spike = resp[15]
                self.n_cont = resp[17]
                gotServerArea = True

        self._init = True
        logger.info("Connection established!")
        
    def select_spikes(self, channels=None, waveforms=True):
        '''Sets the channels from which to receive spikes. This function always requests sorted data

        Parameters
        ----------
        channels : array_like, optional
            A list of channels which you want to see spikes from
        waveforms : bool, optional
            Request spikes from all selected channels
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
        if channels is None:
            raw += array.array('b', [bitmask]*(PACKETSIZE - 20)).tostring()
        else:
            packet = array.array('b', '\x00'*(PACKETSIZE - 20))
            for c in channels:
                packet[c] = bitmask
            raw += packet.tostring()

        self.sock.sendall(raw)
    
    def select_continuous(self, channels=None):
        '''Sets the channels from which to receive continuous data'''
        if not self._init:
            raise ValueError("Please initialize the connection first")
        if not self.supports_spikes:
            raise ValueError("Server does not support continuous data streaming!")
        packet = array.array('i', '\x00'*20)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_SELECT_CONTINUOUS_CHANNELS
        packet[2] = 1
        packet[3] = self.n_cont
        raw = packet.tostring()

        if channels is None:
            raw += array.array('b', [1]*(PACKETSIZE - 20)).tostring()
        else:
            packet = array.array('b', '\x00'*(PACKETSIZE - 20))
            for c in channels:
                packet[c] = bitmask
            raw += packet.tostring()
        
        self.sock.sendall(raw)

    def start_data(self):
        '''Start the data pump from plexnet remote'''
        if not self._init:
            raise ValueError("Please initialize the connection first")
        packet = array.array('i', '\x00'*PACKETSIZE)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_START_DATA_PUMP
        self.sock.sendall(packet.tostring())
        self.streaming = True
        logger.info("Started data pump")

    def stop_data(self):
        '''Stop the data pump from plexnet remote'''
        if not self._init:
            raise ValueError("Please initialize the connection first")
        packet = array.array('i', '\x00'*PACKETSIZE)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_STOP_DATA_PUMP
        self.sock.sendall(packet.tostring())
        self.streaming = False
        logger.info("Stopped data pump")

    def disconnect(self):
        '''Disconnect from the plexnet remote server and close all network sockets'''
        if not self._init:
            raise ValueError("Please initialize the connection first")
        packet = array.array('i', '\x00'*PACKETSIZE)
        packet[0] = self.PLEXNET_COMMAND_FROM_CLIENT_TO_SERVER_DISCONNECT_CLIENT
        self.sock.sendall(packet.tostring())
        self.sock.close()
        logger.info("Disconnected from plexon")
    
    def __del__(self):
        self.disconnect()

    def get_data(self):
        '''A generator which yields packets as they are received'''
        assert self._init, "Please initialize the connection first"
        hnames = 'type,Uts,ts,chan,unit,nwave,nword'.split(',')
        while self.streaming:
            packet = self._recv()
            ibuf = struct.unpack('4i', packet[:16])
            if ibuf[0] != 1:
                yield None
            
            self.num_server_dropped = ibuf[2]
            self.num_mmf_dropped = ibuf[3]
            packet = packet[16:]
            
            while len(packet) > 16:
                header = dict(zip(hnames, struct.unpack('hHI4h', packet[:16])))
                packet = packet[16:]
                
                if header['type'] in [0, -1]:
                    yield None
                else:
                    wavedat = None
                    if header['nwave'] > 0:
                        l = header['nwave'] * header['nword'] * 2
                        wavedat = array.array('h', packet[:l])
                        packet = packet[l:]
                    
                    ts = long(header['Uts']) << 32 | header['ts']
                    yield WaveData(type=header['type'], chan=header['chan'],
                        unit=header['unit'], ts=ts, waveform = wavedat)

if __name__ == "__main__":
    import time
    #Initialize the connection
    conn = Connection("10.0.0.13", 6000)
    conn.connect(256) #Request all 256 channels
    conn.select_spikes() #Select all spike channels, and get waveforms too
    conn.start_data() #start the data pump

    data = []
    waves = conn.get_data()
    start = time.time()
    while (time.time()-start) < 10:
        data.append(waves.next())

    print "Received %d data packets" % len(data)
    #Stop the connection
    conn.stop_data()
    conn.disconnect()
