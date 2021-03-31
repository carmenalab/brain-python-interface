from libcpp cimport bool

cdef extern from "plexread.h":
    ctypedef struct PL_FileHeader:
        unsigned int MagicNumber
        int     Version
        char    Comment[128]
        int     ADFrequency
        int     NumDSPChannels
        int     NumEventChannels
        int     NumSlowChannels
        int     NumPointsWave
        int     NumPointsPreThr

        int     Year
        int     Month 
        int     Day
        int     Hour
        int     Minute
        int     Second

        int     FastRead
        int     WaveformFreq
        double  LastTimestamp
        
        char    Trodalness
        char    DataTrodalness
        char    BitsPerSpikeSample
        char    BitsPerSlowSample
        unsigned short SpikeMaxMagnitudeMV
        unsigned short SlowMaxMagnitudeMV

        unsigned short SpikePreAmpGain

        char    AcquiringSoftware[18]
        char    ProcessingSoftware[18]

        char    Padding[10]

        int     TSCounts[130][5]
        int     WFCounts[130][5]

        int     EVCounts[512]

    ctypedef struct PL_ChanHeader:
        char Name[32]
        char SIGName[32]
        int Channel
        int WFRate
        int SIG
        int Ref
        int Gain
        int Filter
        int Threshold
        int Method
        int NUnits
        short Template[5][64]
        int Fit[5]
        int SortWidth
        short Boxes[5][2][4]
        int SortBeg
        char Comment[128]
        unsigned char SrcId
        unsigned char reserved
        unsigned short ChanId
        int PadHeader[10]
    
    ctypedef struct PL_SlowChannelHeader:
        char    Name[32]
        int     Channel
        int     ADFreq
        int     Gain
        int     Enabled
        int     PreAmpGain

    ctypedef enum ChanType:
        spike, event, wideband, spkc, lfp, analog

    ctypedef struct PlexFile:
        double length
        int nchans[6]
        char* filename

        PL_FileHeader header
        PL_ChanHeader* chan_info
        PL_SlowChannelHeader* cont_head

    ctypedef struct Spike:
        pass

    ctypedef struct ContInfo:
        PlexFile* plxfile
        ChanType type
        unsigned long len
        unsigned long nchans
        double t_start
        int freq
        double start
        double stop
        long long _fedge[2]

    ctypedef struct SpikeInfo:
        PlexFile* plxfile
        ChanType type
        int num
        short wflen
        unsigned long _fedge[2]

    ctypedef struct IterSpike:
        SpikeInfo* info

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

    void plx_print_frameset(PlexFile* plxfile, ChanType type, int start, int stop, bool detail)
    int plx_check_frames(PlexFile* plxfile, ChanType type)
    void plx_summary(PlexFile* plx)
