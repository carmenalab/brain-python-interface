#ifndef _PLEXFILE_H
#define _PLEXFILE_H

#include "dataframe.h"

typedef struct {
    FILE* fp;
    PL_FileHeader header;
    PL_ChanHeader chan_info[MAX_SPIKE_CHANNELS];
    PL_EventHeader event_info[MAX_EVENT_CHANNELS];
    PL_SlowChannelHeader cont_info[MAX_SLOW_CHANNELS];

    FrameSet spikes;
    FrameSet wideband;
    FrameSet spkc;
    FrameSet lfp;
    FrameSet analog;
    FrameSet event;

    unsigned long nframes;
    char* filename;
} PlexFile;

PlexFile* open_plex(char* filename);
void save_cache(PlexFile* plxfile);
long get_header(PlexFile* plxfile);
char* _plx_cache_name(char* filename);

#endif