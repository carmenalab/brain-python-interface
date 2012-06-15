import numpy as np
import ctypes as C

plexlib = C.LoadLibrary("./plexfile.so")
class SpikeData(C.Structure):
    _fields_ = [
        ("num", C.c_int), 
        ("wflen", C.c_short), 
        ("spikes", C.POINTER(C.c_char)),
        ("waveforms", C.c_void_p)
    ]
plexlib.plx_open.restype = C.c_void_p
plexlib.plx_read_events_spikes.restype = SpikeData
plexlib.plx_read_events_spikes.argtypes = [C.c_void_p, C.c_int, C.c_double, C.c_double, C.c_bool]

class SpikeEventFrameset(object):
    dtype = np.dtype([("ts", np.float), ("chan", np.int32), ("unit", np.int32)])
    def __init__(self, plxfile, idx):
        self.plxfile = plxfile
        self.idx = idx

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError("Can only slice in time for events and spikes")
        data = plexlib.plx_read_events_spikes(self.plxfile, self.idx, item.start or -1, item.stop or -1, False)
        return np.ctypeslib.as_array(data.spikes).astype(self.dtype)


class PlexFile(object):
    def __init__(self, filename):
        self.plxfile = plexlib.plx_open(filename)
        self.spikes = SpikeEventFrameset(self.plxfile, 0)
        self.events = SpikeEventFrameset(self.plxfile, 1)