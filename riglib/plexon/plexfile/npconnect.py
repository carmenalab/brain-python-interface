import os
import numpy as np
import ctypes as C
from numpy.ctypeslib import ndpointer

plexlib = np.ctypeslib.load_library("plexfile.so", os.path.split(os.path.abspath(__file__))[0])

SpikeType = np.dtype([("ts", np.float), ("chan", np.int32), ("unit", np.int32)])

class SpikeInfo(C.Structure):
    _fields_ = [
        ("num", C.c_int), 
        ("wflen", C.c_short), 
        ("start", C.c_double),
        ("stop", C.c_double),
    ]

class ContInfo(C.Structure):
    _fields_ = [
        ("len", C.c_ulong), 
        ("nchans", C.c_ulong), 
        ("t_start", C.c_double),
        ("start", C.c_double),
        ("stop", C.c_double),
        ("freq", C.c_int), 
    ]

class PlexFile(C.Structure):
    _fields_ = [
        ("length", C.c_double),
        ("nchans", C.c_int*6)
    ]

plexlib.plx_open.restype = C.POINTER(PlexFile)
plexlib.plx_close.argtypes = [C.POINTER(PlexFile)]

plexlib.plx_get_continuous.restype = C.POINTER(ContInfo)
plexlib.plx_get_continuous.argtypes = [C.POINTER(PlexFile), C.c_int, C.c_double, C.c_double, C.POINTER(C.c_int), C.c_int]
plexlib.plx_read_continuous.argtypes = [C.POINTER(ContInfo), ndpointer(dtype=float, ndim=2, flags='contiguous')]
plexlib.free_continfo.argtypes = [C.POINTER(ContInfo)]

plexlib.plx_get_discrete.restype = C.POINTER(SpikeInfo)
plexlib.plx_get_discrete.argtypes = [C.POINTER(PlexFile), C.c_int, C.c_double, C.c_double]
plexlib.plx_read_discrete.argtypes = [C.POINTER(SpikeInfo), ndpointer(dtype=SpikeType, flags='contiguous')]
plexlib.plx_read_waveforms.argtypes = [C.POINTER(SpikeInfo), ndpointer(dtype=float, ndim=2, flags='contiguous')]
plexlib.free_spikeinfo.argtypes = [C.POINTER(SpikeInfo)]

class DiscreteFrameset(object):
    def __init__(self, plxfile, idx):
        self.plxfile = plxfile
        self.idx = idx

        class WFGet(object):
            def __getitem__(self, item):
                if not isinstance(item, slice):
                    raise TypeError("Can only slice in time for events and spikes")
        self.waveforms = WFGet()

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError("Can only slice in time for events and spikes")
        info = plexlib.plx_get_discrete(self.plxfile, self.idx, item.start or -1, item.stop or -1)
        
        data = np.empty(info.contents.num, dtype=SpikeType)
        plexlib.plx_read_discrete(info, data)
        plexlib.free_spikeinfo(info)
        return data

class ContinuousFrameset(object):
    def __init__(self, plxfile, idx):
        self.plxfile = plxfile
        self.idx = idx
        self.nchans = 0
        self.nchans = plxfile.contents.nchans[idx];

    def __getitem__(self, item):
        chans = None
        nchans = 0
        if isinstance(item, slice):
            start, stop = item.start or -1, item.stop or -1
        elif isinstance(item, tuple):
            start, stop = item[0].start or -1, item[0].stop or -1
            if isinstance(item[1], slice):
                chans = range(*item[1].indices(self.nchans))
            elif isinstance(item[1], (tuple, list, np.ndarray)):
                chans = item[1]
            elif isinstance(item, int):
                chans = [item[1]]
        else:
            raise TypeError("Invalid slice")

        if chans is not None:
            nchans = len(chans)
            chans = (C.c_int*len(chans))(*chans)

        print chans, nchans
        info = plexlib.plx_get_continuous(self.plxfile, self.idx, start, stop, chans, nchans)
        if info.contents.t_start != 0:
            print "Time offset: %f"%info.contents.t_start
        data = np.empty((info.contents.len, info.contents.nchans))
        plexlib.plx_read_continuous(info, data)
        plexlib.free_continfo(info)
        return data


class PlexData(object):
    def __init__(self, filename):
        self.plxfile = plexlib.plx_open(filename)
        self.spikes = DiscreteFrameset(self.plxfile, 0)
        self.events = DiscreteFrameset(self.plxfile, 1)
        self.wideband = ContinuousFrameset(self.plxfile, 2)
        self.spkc = ContinuousFrameset(self.plxfile, 3)
        self.lfp = ContinuousFrameset(self.plxfile, 4)
        self.analog = ContinuousFrameset(self.plxfile, 5)

    def summary(self):
        plexlib.plx_summary(self.plxfile)

    def __del__(self):
        plexlib.plx_close(self.plxfile)

    def __len__(self):
        return self.plxfile.contents.length

if __name__ == "__main__":
    plx = PlexData("/tmp/dat03062012003.plx")
    data = plx.wideband[20:22.5, 161]
    print data