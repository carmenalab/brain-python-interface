#ifndef _PSTH_H
#define _PSTH_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "spike.h"

#define SPIKE_FUZZ 0.05
#define BUF_INIT_SIZE 1

typedef unsigned int uint;
typedef unsigned short ushort;

typedef struct Channel {
    int chan;
    int unit;
} Channel;

typedef struct SpikeBuf {
	Spike* data;
	uint size;
	unsigned long idx;
} SpikeBuf;

typedef struct BinInfo {
	uint nunits;
	double binlen;

	uint chanmap[8192];
	double (*binfunc)(double, double, double*);
	double params[32];
} BinInfo;

typedef struct BinInc {
	double* times;
	uint tlen;

	BinInfo* info;
	SpikeBuf spikes;
	uint _tidx;
} BinInc;

unsigned int _hash_chan(ushort chan, ushort unit);
double boxcar(double start, double ts, double* params);
double gaussian(double start, double ts, double* params);

extern BinInfo* bin_init(int* chans, size_t clen, double binlen, char* funcname, double* params);
extern void bin_spikes(BinInfo* info, Spike* spikes, uint nspikes, double* output);

extern BinInc* bin_incremental(BinInfo* info, double* times, uint tlen);
extern int bin_inc_spike(BinInc* inc, Spike* spike, double* output);

extern void free_bininfo(BinInfo* info);
extern void free_bininc(BinInc* inc);

#endif