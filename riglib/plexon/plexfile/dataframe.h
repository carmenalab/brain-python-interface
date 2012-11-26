#ifndef _DATAFRAME_H
#define _DATAFRAME_H
#include "plexfile.h"

#define ChanType_MAX (6)

struct PlexFile;
typedef struct PlexFile PlexFile;
typedef unsigned long long TSTYPE;

typedef enum {
    spike, event, wideband, spkc, lfp, analog
} ChanType;

typedef struct SimpleDatablock {
    ChanType type;
    TSTYPE ts;
    short chan;
    short unit;
    short samples;
} SimpleDatablock;

typedef struct DataFrame {
    TSTYPE ts;
    size_t fpos;
    unsigned short samples;
    unsigned short nblocks;
} DataFrame;

typedef struct FrameSet {
    DataFrame* frames;
    unsigned long lim;
    unsigned long num;
} FrameSet;

void plx_get_frames(PlexFile* plxfile);
long int _plx_read_datablock(FILE* fp, int nchannels, SimpleDatablock* block);
DataFrame* _plx_new_frame(SimpleDatablock* header, unsigned long start, PlexFile* plxfile);

#endif
