import numpy as np
cimport numpy as np
cimport cpsth

cdef class BinInfo:
    cdef cpsth.BinInfo *info
    
    def __cinit__(self, np.ndarray[np.uint32_t, ndim=2] channels, double binlen, funcname='boxcar', params=None):
        cdef double _params[32]
        if params is not None:
            _params = <double*> params.data

        self.info = cpsth.bin_init(channels.data, len(channels.data), binlen, funcname, _params)
        if self.info is NULL:
            raise MemoryError

