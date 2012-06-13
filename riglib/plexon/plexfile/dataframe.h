#ifndef _DATAFRAME_H
#define _DATAFRAME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include "Plexon.h"

#define min(a,b) (((a) < (b)) ? (a) : (b))

#define MAX_SPIKE_CHANNELS   (256)
#define MAX_EVENT_CHANNELS   (512)
#define MAX_SLOW_CHANNELS    (1024)
#define MAX_SAMPLES_PER_WAVEFORM (256)
#define STROBED_CHANNEL 257

typedef struct PL_FileHeader PL_FileHeader;
typedef struct PL_ChanHeader PL_ChanHeader;
typedef struct PL_EventHeader PL_EventHeader;
typedef struct PL_SlowChannelHeader PL_SlowChannelHeader;
typedef struct PL_DataBlockHeader PL_DataBlockHeader;
typedef unsigned char uchar;


#define chantype_spike      (1)
#define chantype_wideband   (2)
#define chantype_spkc       (3)
#define chantype_lfp        (4)
#define chantype_analog     (5)
#define chantype_event      (6)

typedef struct {
    short type;
    unsigned long long ts;
    short chan;
    short unit;
    short samples;
} SimpleDatablock;

typedef struct {
    long long ts;
    short type;
    long int samples;
    size_t fpos[2];
    unsigned long nblocks;
} DataFrame;

typedef struct {
    DataFrame* frames;
    unsigned long lim;
    unsigned long num;
} FrameSet;

#endif