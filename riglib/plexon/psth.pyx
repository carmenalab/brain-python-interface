import numpy as np
cimport numpy as np
np.import_array()

from plexon cimport psth

cdef class SpikeBin:
    def __cinit__(self, object channels, double binlen=0.1, funcname='boxcar', params=None):
        cdef double* _params = NULL
        cdef np.ndarray[np.double_t] np_param
        cdef np.ndarray _chan

        if funcname == 'gaussian':
            assert params is not None
            np_param = np.zeros((2,), dtype=np.double)
            np_param[:] = params['mean'], params['std']
            _params = <double*> np_param.data
        
        _chan = np.ascontiguousarray(channels, dtype=np.int32)
        self.nunits = len(channels)

        self.info = psth.bin_init(<int*> _chan.data, self.nunits, binlen, funcname, _params)
        if self.info is NULL:
            raise MemoryError

    def __dealloc__(self):
        if self.info is not NULL:
            psth.free_bininfo(self.info)

    def __call__(self, np.ndarray spikes):
        cdef np.ndarray[np.double_t, ndim=1] output = np.zeros((self.info.nunits,), dtype=np.double)
        psth.bin_spikes(self.info, <psth.Spike*> spikes.data, len(spikes), <double*> output.data)
        return output