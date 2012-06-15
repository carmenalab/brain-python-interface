import numpy as np
import ctypes as C

plexlib = C.cdll.LoadLibrary("./plexfile.so")

class Spike(C.Structure):
    _fields_ = [
        ("ts", C.c_double),
        ("chan", C.c_int),
        ("unit", C.c_int),
    ]

class SpikeData(C.Structure):
    _fields_ = [
        ("num", C.c_int), 
        ("wflen", C.c_short), 
        ("spikes", C.POINTER(Spike)),
        ("waveforms", C.POINTER(C.c_double))
    ]

class ContData(C.Structure):
    _fields_ = [
        ("len", C.c_ulong), 
        ("nchans", C.c_ulong), 
        ("t_start", C.c_double),
        ("freq", C.c_int), 
        ("data", C.POINTER(C.c_double)),
    ]

plexlib.plx_open.restype = C.c_void_p
plexlib.plx_read_events_spikes.restype = C.POINTER(SpikeData)
plexlib.plx_read_events_spikes.argtypes = [C.c_void_p, C.c_int, C.c_double, C.c_double, C.c_bool]
plexlib.plx_read_continuous.restype = C.POINTER(ContData)
plexlib.plx_read_continuous.argtypes = [C.c_void_p, C.c_int, C.c_double, C.c_double, C.POINTER(C.c_int), C.c_int]

class SpikeEventFrameset(object):
    dtype = np.dtype([("ts", np.float), ("chan", np.int32), ("unit", np.int32)])
    def __init__(self, plxfile, idx):
        self.plxfile = plxfile
        self.idx = idx

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError("Can only slice in time for events and spikes")
        spikes = plexlib.plx_read_events_spikes(self.plxfile, self.idx, item.start or -1, item.stop or -1, False)
        data = np.ctypeslib.as_array(spikes.contents.spikes, shape=(spikes.contents.num,))
        data.dtype = self.dtype
        return data

    @property
    def waveforms(self):
        _self = self
        class WFGet(object):
            def __getitem__(self, item):
                if not isinstance(item, slice):
                    raise TypeError("Can only slice in time for events and spikes")


class ContinuousFrameset(object):
    def __init__(self, plxfile, idx):
        self.plxfile = plxfile
        self.idx = idx

    def __getitem__(self, item):
        if isinstance(item, slice):
            cont = plexlib.plx_read_continuous(self.plxfile, self.idx, item.start or -1, item.stop or -1, None, 0)
            data = np.ctypeslib.as_array(cont.contents.data, shape=(cont.contents.len, cont.contents.nchans))
            return data


class PlexFile(object):
    def __init__(self, filename):
        self.plxfile = plexlib.plx_open(filename)
        self.spikes = SpikeEventFrameset(self.plxfile, 0)
        self.events = SpikeEventFrameset(self.plxfile, 1)
        self.analog = ContinuousFrameset(self.plxfile, 5)

    def summary(self):
        plexlib.plx_summary(self.plxfile)