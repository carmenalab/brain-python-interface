#ifndef _PLEXREAD_H_
#define _PLEXREAD_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#include "plexfile.h"
#include "dataframe.h"
typedef unsigned long long TSTYPE;

typedef struct {
    unsigned long len;
    unsigned long nchans;
    int t_start;
    int freq;
    double* data;
} ContData;

typedef struct {
    double ts;
    int chan;
    int unit;
} Spike;

typedef struct {
    int num;
    short wflen;
    Spike* spike;
    double* waveforms;
} SpikeData;

void plx_read_continuous(FILE* fp, FrameSet* frameset, int tsfreq, int chanfreq, int gain,
    TSTYPE start, TSTYPE stop, int* chans, int nchans, 
    double* data);
ContData* plx_readall_analog(PlexFile* plxfile);


#endif