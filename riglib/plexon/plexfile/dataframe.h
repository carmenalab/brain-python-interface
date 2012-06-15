#ifndef _DATAFRAME_H
#define _DATAFRAME_H

#include "plexfile.h"

#define ChanType_MAX (6)

typedef struct PlexFileType PlexFile;
typedef unsigned long long TSTYPE;

typedef enum {
    spike, event, wideband, spkc, lfp, analog
} ChanType;

typedef struct {
    ChanType type;
    TSTYPE ts;
    short chan;
    short unit;
    short samples;
} SimpleDatablock;

typedef struct DataFrameType {
    TSTYPE ts;
    ChanType type;
    size_t fpos[2];
    unsigned long samples;
    unsigned long nblocks;
} DataFrame;

typedef struct FrameSetType {
    DataFrame* frames;
    unsigned long lim;
    unsigned long num;
} FrameSet;

void plx_get_frames(PlexFile* plxfile);
long int _plx_read_datablock(FILE* fp, int nchannels, SimpleDatablock* block);
void _plx_new_frame(SimpleDatablock* header, unsigned long start, PlexFile* plxfile, DataFrame** frame);

#endif