#ifndef _PLEXREAD_H_
#define _PLEXREAD_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>

#include "plexfile.h"

typedef struct ContDataType {
    unsigned long len;
    unsigned long nchans;
    double t_start;
    int freq;
    double* data;
} ContData;

typedef struct SpikeType {
    double ts;
    int chan;
    int unit;
} Spike;

typedef struct SpikeDataType {
    int num;
    short wflen;
    Spike* spike;
    double* waveforms;
} SpikeData;

ContData* plx_read_continuous(PlexFile* plxfile, ChanType type,
    double start, double stop, int* chans, int nchans);
SpikeData* plx_readall_spikes(PlexFile* plxfile, bool waveforms);

unsigned long _binary_search(FrameSet* frameset, TSTYPE ts);

#endif