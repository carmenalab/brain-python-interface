cimport numpy as np

cdef extern from "psth.h":
    ctypedef unsigned short ushort
    ctypedef unsigned int uint
    
    ctypedef struct BinInfo:
        uint nunits
        double binlen

    ctypedef struct BinInc:
        pass
    ctypedef struct Channel:
        pass
    ctypedef struct Spike:
        pass

    BinInfo* bin_init(int* chans, size_t clen, double binlen, char* funcname, double* params)
    void bin_spikes(BinInfo* info, Spike* spikes, uint nspikes, double* output)

    BinInc* bin_incremental(BinInfo* info, double* times, uint tlen)
    bint bin_inc_spike(BinInc* inc, Spike* spike)

    extern void free_bininfo(BinInfo* info)
    extern void free_bininc(BinInc* inc)
