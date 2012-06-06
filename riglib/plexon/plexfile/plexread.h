#ifndef _PLEXREAD_H_
#define _PLEXREAD_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define false 0

#include "Plexon.h"

#define MAX_SPIKE_CHANNELS   (256)
#define MAX_EVENT_CHANNELS   (512)
#define MAX_SLOW_CHANNELS    (1024)
#define MAX_SAMPLES_PER_WAVEFORM (256)
#define STROBED_CHANNEL 257
#define MAX_SPIKES_PER_ELECTRODE 5

typedef struct PL_FileHeader PL_FileHeader;
typedef struct PL_ChanHeader PL_ChanHeader;
typedef struct PL_EventHeader PL_EventHeader;
typedef struct PL_SlowChannelHeader PL_SlowChannelHeader;
typedef struct PL_DataBlockHeader PL_DataBlockHeader;
typedef unsigned char uchar;

#endif