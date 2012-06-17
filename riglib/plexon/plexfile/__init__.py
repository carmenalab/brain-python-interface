import os
import numpy as np
import ctypes as C
from numpy.ctypeslib import ndpointer

import finalizer
import Plexon_h

cwd = os.path.split(os.path.abspath(__file__))[0]
if not os.path.isfile(os.path.join(cwd, "plexfile.so")):
    import subprocess as sp
    sp.Popen(["make", "lib"], cwd=cwd).wait()

plexlib = np.ctypeslib.load_library("plexfile.so", cwd)

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
        ("nchans", C.c_int*6),
        ("filename", C.c_char_p),
        ("fp", C.c_void_p),
        ("header", Plexon_h.PL_FileHeader),
        ("chan_info", Plexon_h.PL_ChanHeader*256),
        ("event_info", Plexon_h.PL_EventHeader*256),
        ("cont_head", Plexon_h.PL_SlowChannelHeader*1024),
        ("cont_info", C.POINTER(Plexon_h.PL_SlowChannelHeader)*4)
    ]

SpikeType = np.dtype([("ts", np.float), ("chan", np.int32), ("unit", np.int32)])

plexlib.plx_open.restype = C.POINTER(PlexFile)
plexlib.plx_open.argtypes = [C.c_char_p, C.c_bool]
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
                info = plexlib.plx_get_discrete(plxfile, idx, item.start or -1, item.stop or -1)
                data = np.empty((info.contents.num, info.contents.wflen))
                plexlib.plx_read_waveforms(info, data)
                plexlib.free_spikeinfo(info)
                return data

        self.waveforms = WFGet()

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError("Can only slice in time for events and spikes")
        info = plexlib.plx_get_discrete(self.plxfile, self.idx, item.start or -1, item.stop or -1)
        
        data = np.empty(info.contents.num, dtype=SpikeType)
        plexlib.plx_read_discrete(info, data)
        plexlib.free_spikeinfo(info)
        return data

class ContinuousSlice(object):
    def __init__(self, info):
        self.info = info
        finalizer.track_for_finalization(self, info, self._free)

    @property
    def data(self):
        data = np.empty((self.info.contents.len, self.info.contents.nchans))
        plexlib.plx_read_continuous(self.info, data)
        return data

    @property
    def time(self):
        return np.arange(self.info.contents.t_start, self.info.contents.stop, 1./self.info.contents.freq)

    def __repr__(self):
        return "<Continuous slice t=[%f:%f], %d channels>"%(self.info.contents.start, self.info.contents.stop, self.info.contents.nchans)

    def __len__(self):
        return self.info.contents.len

    def _free(self):
        print "Deleting continuous slice %r"%self.info
        plexlib.free_continfo(self.info)

class ContinuousFrameset(object):
    def __init__(self, plxfile, idx):
        self.plxfile = plxfile
        self.idx = idx
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
            elif isinstance(item[1], int):
                chans = [item[1]]
            else:
                raise TypeError("Invalid channel selection")
        else:
            raise TypeError("Invalid time slice")

        if chans is not None:
            nchans = len(chans)
            chans = (C.c_int*len(chans))(*chans)

        info = plexlib.plx_get_continuous(self.plxfile, self.idx, start, stop, chans, nchans)
        return ContinuousSlice(info)

class DataFile(object):
    def __init__(self, filename, recache=False):
        self.plxfile = plexlib.plx_open(filename, recache)
        finalizer.track_for_finalization(self, self.plxfile, self._free)

        self.spikes = DiscreteFrameset(self.plxfile, 0)
        self.events = DiscreteFrameset(self.plxfile, 1)
        self.wideband = ContinuousFrameset(self.plxfile, 2)
        self.spkc = ContinuousFrameset(self.plxfile, 3)
        self.lfp = ContinuousFrameset(self.plxfile, 4)
        self.analog = ContinuousFrameset(self.plxfile, 5)

    def summary(self):
        plexlib.plx_summary(self.plxfile)

    def check(self):
        for i in range(6):
            plexlib.plx_check_frames(self.plxfile, i)

    def __len__(self):
        return self.plxfile.contents.length

    def _free(self):
        print "Deleting plexfile object %r"%self.plxfile
        plexlib.plx_close(self.plxfile)

def openFile(filename, recache=False):
    plx = DataFile(filename, recache)
    plx.summary()
    return plx
