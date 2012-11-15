import numpy as np
cimport numpy as np
cimport cpsth

cdef class SpikeBin:
    cdef cpsth.BinInfo* info
    
    def __cinit__(self, np.ndarray[np.int32_t, ndim=2] channels, double binlen, funcname='boxcar', params=None):

        cdef double* _params
        cdef np.ndarray[np.double_t] np_param

        if funcname == 'gaussian':
            assert params is not None
            np_param = np.zeros((2,), dtype=np.double)
            np_param[:] = params['mean'], params['std']
            _params = <double*> np_param.data

        self.info = cpsth.bin_init(<int*> channels.data, len(channels), binlen, funcname, _params)
        if self.info is NULL:
            raise MemoryError

    def __dealloc__(self):
        if self.info is not NULL:
            cpsth.free_bininfo(self.info)

    def __call__(self, np.ndarray spikes):
        cdef np.ndarray[np.double_t, ndim=1] output = np.zeros((self.info.nunits,), dtype=np.double)
        cpsth.bin_spikes(self.info, <cpsth.Spike*> spikes.data, len(spikes), <double*> output.data)
        return output