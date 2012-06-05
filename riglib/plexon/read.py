import ctypes
from plexfile_header import PL_FileHeader, PL_ChanHeader, PL_EventHeader, PL_SlowChannelHeader

class Plexfile(object):
    def __init__(self, filename):
        self.file = open(filename, "r")
        self.header = PL_FileHeader()
        hsize = ctypes.sizeof(PL_FileHeader)
        ctypes.memmove(ctypes.addressof(self.header), self.file.read(hsize), hsize)

        ch_size = ctypes.sizeof(PL_ChanHeader) * self.header.NumDSPChannels
        ev_size = ctypes.sizeof(PL_EventHeader) * self.header.NumEventChannels
        sl_size = ctypes.sizeof(PL_SlowChannelHeader) * self.header.NumSlowChannels

        self.channels = (PL_ChanHeader*self.header.NumDSPChannels)()
        self.events = (PL_EventHeader*self.header.NumEventChannels)()
        self.slowchans = (PL_SlowChannelHeader*self.header.NumSlowChannels)()
        ctypes.memmove(ctypes.addressof(self.channels), self.file.read(ch_size), ch_size)
        ctypes.memmove(ctypes.addressof(self.events), self.file.read(ev_size), ev_size)
        ctypes.memmove(ctypes.addressof(self.slowchans), self.file.read(sl_size), sl_size)
        self._dstart = sum([hsize, ch_Size, ev_size, sl_size])