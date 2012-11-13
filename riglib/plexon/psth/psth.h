#ifndef _PSTH_H
#define _PSTH_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define SPIKE_SEARCH 0.05
#define BUF_INIT_SIZE 1024

typedef unsigned int uint;
typedef unsigned short ushort;

typedef struct {
    double ts;
    int chan;
    int unit;
} Spike;

typedef struct {
    int chan;
    int unit;
} Channel;

typedef struct {
	Spike* data;
	uint size;
	uint idx;
} SpikeBuf;

typedef struct {
	uint nunits;
	double binlen;

	uint chanmap[8192];
	double (*binfunc)(double, double, double, double*);
	double params[32];
} BinInfo;

typedef struct {
	double output[8192];
	double* times;
	uint tlen;

	BinInfo* info;
	SpikeBuf spikes;
	bool _reset;
	uint _tidx;
} BinInc;

unsigned int _hash_chan(ushort chan, ushort unit);
double boxcar(double start, double ts, double* params);
double gaussian(double start, double ts, double* params);

extern BinInfo* bin_init(char* bufchan, size_t clen, double binlen, char* funcname);
extern void bin_spikes(BinInfo* info, Spike* spikes, uint nspikes, double* output);

extern BinInc* bin_incremental(BinInfo* info, double* times, uint tlen);
extern bool bin_inc_spike(BinInc* inc, Spike* spike);
extern bin_inc_get(BinInc* inc, double* data);

#endif