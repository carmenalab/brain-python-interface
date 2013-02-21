cimport numpy as np
import os
import numpy as np
from libcpp cimport bool

from plexon cimport *

spiketype = [('ts', np.double),('chan', np.int32),('unit', np.int32)]

cdef class ContinuousFS
cdef class Continuous
cdef class DiscreteFS
cdef class Discrete

cdef class Datafile:
    cdef PlexFile* plxfile
    cdef public DiscreteFS spikes
    cdef public DiscreteFS events
    cdef public ContinuousFS wideband
    cdef public ContinuousFS spkc
    cdef public ContinuousFS lfp
    cdef public ContinuousFS analog

    def __cinit__(self, bytes filename):
        if not os.path.exists(filename):
            raise IOError("File not found")

        self.plxfile = plx_open(filename)
        if self.plxfile is NULL:
            raise MemoryError

    def load(self, bool recache):
        plx_load(self.plxfile, recache)
        self.spikes = DiscreteFS(self, spike)
        self.events = DiscreteFS(self, event)
        self.wideband = ContinuousFS(self, wideband)
        self.spkc = ContinuousFS(self, spkc)
        self.lfp = ContinuousFS(self, lfp)
        self.analog = ContinuousFS(self, analog)
        plx_summary(self.plxfile)

    def __dealloc__(self):
        if self.plxfile is not NULL:
            plx_close(self.plxfile)

cdef class ContinuousFS:
    cdef Datafile parent
    cdef ChanType type
    cdef unsigned long nchans

    def __cinit__(self, Datafile parent, ChanType type):
        self.type = type
        self.parent = parent
        self.nchans = parent.plxfile.nchans[<int>type]

    def __getitem__(self, item):
        cdef double start, stop

        if isinstance(item, slice):
            start, stop = item.start or -1, item.stop or -1
            chans = slice(None)
        elif isinstance(item, tuple):
            start, stop = item[0].start or -1, item[0].stop or -1
            chans = item[1]
        else:
            raise TypeError("Invalid slice")

        return Continuous(self, start, stop, chans)

cdef class Continuous:
    cdef ContInfo* info

    def __cinit__(self, ContinuousFS fs, double start, double stop, object chans):
        cdef int _chans[1024]
        cdef int nchans = 0
        cdef int i, c

        if isinstance(chans, slice):
            for i, c in enumerate(xrange(*chans.indices(fs.nchans))):
                _chans[i] = c
                nchans += 1
        elif isinstance(chans, (tuple, list, np.ndarray)):
            for i, c in enumerate(chans):
                _chans[i] = c
                nchans += 1
        elif isinstance(chans, int):
            _chans[0] = chans
            nchans = 1
        else:
            raise TypeError("Invalid channel selection")

        self.info = plx_get_continuous(fs.parent.plxfile, fs.type, start, stop, _chans, nchans)
        if self.info is NULL:
            raise MemoryError

    def __dealloc__(self):
        if self.info is not NULL:
            free_continfo(self.info)

    property data:
        def __get__(self):
            cdef np.ndarray data = np.zeros((self.info.len, self.info.nchans), dtype=np.double)
            if plx_read_continuous(self.info, <double*>data.data):
                raise IOError('Error reading plexfile')
            return data
        
    property time:
        def __get__(self):
            cdef double stop = self.info.len / self.info.freq
            return np.arange(self.info.t_start, stop, 1./self.info.freq)

cdef class DiscreteFS:
    cdef Datafile parent
    cdef ChanType type

    def __cinit__(self, Datafile parent, ChanType type):
        self.type = type
        self.parent = parent

    def __getitem__(self, idx):
        assert isinstance(idx, slice), "Discrete channels only support slicing in time"
        cdef double start = -1 if idx.start is None else idx.start
        cdef double stop = -1 if idx.stop is None else idx.stop
        return Discrete(self, start, stop)

cdef class Discrete:
    cdef SpikeInfo* info
    cdef IterSpike* it

    def __cinit__(self, DiscreteFS fs, double start, double stop):
        self.info = plx_get_discrete(fs.parent.plxfile, fs.type, start, stop)
        if self.info is NULL:
            raise MemoryError

        self.it = plx_iterate_discrete(self.info)

    def __dealloc__(self):
        if self.info is not NULL:
            free_spikeinfo(self.info)
        if self.it is not NULL:
            free_iterspike(self.it)

    def __next__(self):
        cdef np.ndarray data = np.empty((1,), dtype=spiketype)
        cdef int status = plx_iterate(self.it, <Spike*> data.data)
        while status == 0:
            status = plx_iterate(self.it, <Spike*> data.data)
        if status == 2:
            raise StopIteration
        elif status < 0:
            raise IOError('Error reading plexfile')

        return data

    def __iter__(self):
        return self

    property data:
        def __get__(self):
            cdef np.ndarray data = np.zeros((self.info.num,), dtype=spiketype)
            if plx_read_discrete(self.info, <Spike*> data.data):
                raise IOError('Error reading plexfile')
            return data

    property waveforms:
        def __get__(self):
            cdef np.ndarray data = np.zeros((self.info.num, self.info.wflen), dtype=np.double)
            if plx_read_waveforms(self.info, <double*> data.data):
                raise IOError('Error reading plexfile')
            return data


def openFile(bytes filename, bool load=True, bool recache=False):
    plx = Datafile(filename)
    if load:
        plx.load(recache)
    return plx