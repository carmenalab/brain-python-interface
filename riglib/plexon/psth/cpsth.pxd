cdef extern from "psth.h":
    ctypedef struct BinInfo:
        pass
    ctypedef struct BinInc:
        pass
    ctypedef struct Channel:
        pass
    ctypedef struct Spike:
        pass

    ctypedef unsigned short ushort
    ctypedef unsigned int uint

    BinInfo* bin_init(char* bufchan, size_t clen, double binlen, char* funcname, double* params)
    void bin_spikes(BinInfo* info, Spike* spikes, uint nspikes, double* output)

    BinInc* bin_incremental(BinInfo* info, double* times, uint tlen)
    bint bin_inc_spike(BinInc* inc, Spike* spike)
    void bin_inc_get(BinInc* inc, double* data, double* ts)
