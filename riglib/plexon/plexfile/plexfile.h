#ifndef _PLEXFILE_H
#define _PLEXFILE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include "Plexon.h"

#include "dataframe.h"

#define min(a,b) (((a) < (b)) ? (a) : (b))

#define MAX_SPIKE_CHANNELS   (256)
#define MAX_EVENT_CHANNELS   (512)
#define MAX_SLOW_CHANNELS    (1024)
#define MAX_SAMPLES_PER_WAVEFORM (256)
#define STROBED_CHANNEL 257

typedef unsigned char uchar;
typedef struct PL_FileHeader PL_FileHeader;
typedef struct PL_ChanHeader PL_ChanHeader;
typedef struct PL_EventHeader PL_EventHeader;
typedef struct PL_SlowChannelHeader PL_SlowChannelHeader;
typedef struct PL_DataBlockHeader PL_DataBlockHeader;

typedef struct PlexFileType {
    char* filename;
    FILE* fp;
    PL_FileHeader header;
    PL_ChanHeader chan_info[MAX_SPIKE_CHANNELS];
    PL_EventHeader event_info[MAX_EVENT_CHANNELS];
    PL_SlowChannelHeader cont_head[MAX_SLOW_CHANNELS];
    PL_SlowChannelHeader* cont_info[4];

    FrameSet data[6];
    unsigned long nframes;
} PlexFile;

extern PlexFile* plx_open(char* filename);
extern void plx_close(PlexFile* plxfile);

void plx_save_cache(PlexFile* plxfile);
long plx_get_header(PlexFile* plxfile);
char* _plx_cache_name(char* filename);

#endif