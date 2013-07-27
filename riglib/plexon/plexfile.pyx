import os
import numpy as np
cimport numpy as np
np.import_array()

from libcpp cimport bool

from plexon cimport psth
from plexfile cimport *

spiketype = [('ts', np.double), ('chan', np.int32), ('unit', np.int32)]

cdef class ContinuousFS
cdef class Continuous
cdef class DiscreteFS
cdef class Discrete

cdef class Datafile:
    cdef PlexFile* plxfile
    cdef public double length

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

        self.length = self.plxfile.length

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

    property units:
        def __get__(self):
            cdef int i, j
            cdef object units = []

            for i in range(self.plxfile.header.NumDSPChannels):
                for j in range(self.plxfile.chan_info[i].NUnits):
                    units.append((i+1, j+1))

            return units

    property gain:
        def __get__(self):
            cdef int i, j
            cdef object gains = []
            for i in range(self.plxfile.header.NumDSPChannels):
                gains.append(self.plxfile.chan_info[i].Gain)
            return gains

cdef class ContinuousFS:
    cdef Datafile parent
    cdef ChanType type
    cdef unsigned long nchans

    def __cinit__(self, Datafile parent, ChanType type):
        self.type = type
        self.parent = parent
        self.nchans = parent.plxfile.nchans[<int>type]

    def check(self):
        print plx_check_frames(self.parent.plxfile, self.type)

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
    cdef object shape

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

        self.shape = (self.info.len, self.info.nchans)

    def __dealloc__(self):
        if self.info is not NULL:
            free_continfo(self.info)

    def __len__(self):
        return self.info.len

    def inspect(self):
        plx_print_frameset(self.info.plxfile, self.info.type, self.info._fedge[0], self.info._fedge[1], False)

    property data:
        def __get__(self):
            cdef np.ndarray data = np.zeros(self.shape, dtype=np.double)
            if plx_read_continuous(self.info, <double*>data.data):
                raise IOError('Error reading plexfile')
            return data
        
    property time:
        def __get__(self):
            cdef double start = self.info.start + self.info.t_start
            cdef double stop = start + self.info.len / float(self.info.freq)
            return np.arange(start, stop, 1./self.info.freq)

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

    def bin(self, object[np.double_t, ndim=1] times, psth.SpikeBin sbin=None, double binlen=0.1):
        cdef Discrete spikes

        if sbin is not None:
            binlen = sbin.info.binlen
        
        spikes = Discrete(self, times[0]-binlen-0.05, times[-1]+0.05)
        return IterBin(spikes, sbin, times, binlen)

cdef class Discrete:
    cdef SpikeInfo* info
    cdef Datafile data

    def __cinit__(self, DiscreteFS fs, double start, double stop):
        self.data = fs.parent
        self.info = plx_get_discrete(fs.parent.plxfile, fs.type, start, stop)
        if self.info is NULL:
            raise MemoryError

    def __dealloc__(self):
        if self.info is not NULL:
            free_spikeinfo(self.info)

    def __iter__(self):
        return IterDiscrete(self)

    def __len__(self):
        return self.info.num

    def inspect(self, bool detail=False):
        plx_print_frameset(self.info.plxfile, self.info.type, self.info._fedge[0], self.info._fedge[1], detail)

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

cdef class IterDiscrete:
    cdef Discrete parent
    cdef IterSpike* it

    def __cinit__(self, Discrete parent):
        self.parent = parent
        self.it = plx_iterate_discrete(parent.info)
        if self.it is NULL:
            raise MemoryError

    def __dealloc__(self):
        if self.it is not NULL:
            free_iterspike(self.it)

    def __iter__(self):
        return self

    def __next__(self):
        cdef np.ndarray data = np.empty((1,), dtype=spiketype)
        cdef int status = plx_iterate(self.it, <Spike*> data.data)
        if status == 1:
            raise StopIteration
        elif status < 0:
            raise IOError('Error reading plexfile')
        elif status == 0:
            return data
        else:
            raise Exception('Unknown status response %d'%status)

cdef class IterBin:
    cdef Discrete parent
    cdef psth.SpikeBin bin
    cdef IterSpike* it
    cdef psth.BinInc* inc

    cdef public object shape
    cdef int nunits

    def __cinit__(self, Discrete parent, psth.SpikeBin spikebin, object[np.double_t, ndim=1] times, double binlen=0.1):
        cdef np.ndarray _times
        self.parent = parent
        self.it = plx_iterate_discrete(parent.info)
        if self.it is NULL:
            raise MemoryError

        _times = np.ascontiguousarray(times)
        if spikebin is None:
            spikebin = psth.SpikeBin(parent.data.units, binlen)

        self.bin = spikebin
        self.shape = len(times), self.bin.nunits
        self.nunits = self.bin.nunits

        self.inc = psth.bin_incremental(self.bin.info, <double*> _times.data, len(times))
        if self.inc is NULL:
            raise MemoryError

    def __dealloc__(self):
        if self.it is not NULL:
            free_iterspike(self.it)

    def __iter__(self):
        return self

    def __len__(self):
        return self.shape[0]

    def __next__(self):
        cdef int status = 0
        cdef Spike spike

        cdef np.ndarray[np.double_t, ndim=1] output = np.zeros((self.nunits,), dtype=np.double)

        while status == 0:
            status = plx_iterate(self.it, &spike)
            binstat = psth.bin_inc_spike(self.inc, &spike, <double*> output.data)
            if binstat == 1:
                return output
            elif binstat == 2:
                raise StopIteration

        if status == 1:
            raise StopIteration

        raise IOError('Status %d'%status)

    # def get(self):
    #     cdef int i = 0
    #     cdef int status = 0
    #     cdef Spike spike

    #     cdef np.ndarray[np.double_t, ndim=2] output = np.zeros(self.shape, dtype=np.double)
    #     cdef double* outptr = <double*> (output[i].data)

    #     while status == 0:
    #         status = plx_iterate(self.it, &spike)
    #         if psth.bin_inc_spike(self.inc, &spike, outptr):
    #             i += 1
    #             outptr = <double*> (output[i].data)

    #     return output

def openFile(bytes filename, bool load=True, bool recache=False):
    plx = Datafile(filename)
    if load:
        plx.load(recache)
    return plx
