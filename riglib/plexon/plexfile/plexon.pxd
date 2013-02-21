from libcpp cimport bool

cdef extern from "plexread.h":
	ctypedef struct PlexFile:
		double length
		char* filename
		int nchans[6]

	ctypedef struct Spike:
		pass

	ctypedef struct ContInfo:
		unsigned long len
		unsigned long nchans
		double t_start
		int freq

	ctypedef struct SpikeInfo:
		int num
		short wflen

	ctypedef enum ChanType:
		spike, event, wideband, spkc, lfp, analog

	ctypedef struct IterSpike:
		pass

	PlexFile* plx_open(char* filename)
	void plx_load(PlexFile* plxfile, bool recache)
	void plx_close(PlexFile* plxfile)

	ContInfo* plx_get_continuous(PlexFile* plxfile, ChanType type, double start, double stop, int* chans, int nchans)
	int plx_read_continuous(ContInfo* info, double* data)
	void free_continfo(ContInfo* info)

	SpikeInfo* plx_get_discrete(PlexFile* plxfile, ChanType type, double start, double stop)
	int plx_read_discrete(SpikeInfo* info, Spike* data)
	int plx_read_waveforms(SpikeInfo* info, double* data)
	void free_spikeinfo(SpikeInfo* info)

	IterSpike* plx_iterate_discrete(SpikeInfo* info)
	int plx_iterate(IterSpike* iter, Spike* data)
	void free_iterspike(IterSpike* iter)

	void plx_summary(PlexFile* plx)