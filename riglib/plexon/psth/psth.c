#include "psth.h"

uint _hash_chan(ushort chan, ushort unit) {
    uint ichan = (uint) chan, iunit = (uint) unit;
    return (((ichan & 1023) << 3) | (iunit & 7));
}

double boxcar(double start, double ts, double* params) {
    //First parameter is for box width
    return (double) (start - ts < params[0]);
}

extern BinInfo* bin_init(char* bufchan, size_t clen, double binlen, char* funcname, double* params, uint nparams) {
    int i, idx;
    Channel* channels = (Channel*) bufchan;
    BinInfo* info = (BinInfo*) calloc(1, sizeof(BinInfo));
    info->nunits = clen / sizeof(Channel);
    info->binlen = binlen;

    for (i = 0; i < info->nunits; i++) {
        idx = _hash_chan((ushort) channels[i].chan, (ushort) channels[i].unit);
        info->chanmap[idx] = i+1;
        #ifdef DEBUG
        printf("Setting hash(%d, %d) = %d to %d\n", channels[i].chan, channels[i].unit, idx, i+1);
        #endif
    }

    if (strcmp(funcname, "boxcar") == 0) {
        #ifdef DEBUG
        printf("Using boxcar filter\n");
        #endif
        info->binfunc = &boxcar;
    } else if (strcmp(funcname, "gaussian") == 0) {
        #ifdef DEBUG
        printf("Using gaussian filter\n");
        #endif
        info->binfunc = &gaussian;
    } else {
        return NULL;
    }

    assert(params != NULL);
    assert(nparams < 32);
    memcpy(info->params, params, nparams);

    return info;
}

extern void bin_spikes(BinInfo* info, Spike* spikes, uint nspikes, double* output) {
    uint i, idx, num = slen / sizeof(Spike);
    Spike* spikes = (Spike*) bufspikes;
    double curtime = spikes[num-1].ts, tdiff;

    //Since spike times are only vaguely ordered, find the largest timestamp
    for (i = num-1; i > num-100 && i > 0; i--) {
        if (spikes[i].ts > curtime)
            curtime = spikes[i].ts;
    }

    //Clear the buffer
    for (i = 0; i < countlen; i++) {
        counts[i] = 0;
    }

    //Start at beginning, counting the spikes
    for (i = num-1; i > 0; i--) {
        tdiff = curtime - spikes[i].ts;
        if (tdiff > 1.9*length) {
            //Only exit out when DOUBLE the length of time has been counted (to ensure catching everything)
            //printf("tdiff = %f, curtime=%f, time=%f\n", curtime-spikes[i].ts, curtime, spikes[i].ts);
            return;
        }

        idx = _hash_chan(spikes[i].chan, spikes[i].unit);
        //printf("%d(%d, %d): %d \n", idx, spikes[i].chan, spikes[i].unit, chanmap[idx]);
        if (chanmap[idx] > 0 && chanmap[idx]-1 < countlen && tdiff <= length) {
            counts[chanmap[idx]-1]++;
            //printf("found (%d,%d), incrementing %d\n", spikes[i].chan, spikes[i].unit, chanmap[idx]-1);
        }
    }
    printf("Buffer underrun\n", curtime - spikes[0].ts, length);
}

extern BinInc* bin_incremental(BinInfo* info, double* times, uint tlen) {
    BinInc* inc = (BinInc*) calloc(1, sizeof(BinInc));
    inc->info = info;
    inc->spikes.data = calloc(BUF_INIT_SIZE, sizeof(Spikes));
    inc->spikes.size = BUF_INIT_SIZE;
    inc->times = times;
    inc->tlen = tlen;

    return inc;
}

extern bool bin_inc_spike(BinInc* inc, Spike* spike) {
    uint i, idx, chan;
    Spikebuf* buf = &(inc->spikes);

    Spike* sptr, last = &(buf->data[(buf->idx+1)%buf->size]);
    if (buf->idx < buf->size)
        last = &(buf->data[0]);

    double sdiff = buf->idx == 0 ? 0 : spike->ts - last->ts;

    //If time range in buffer is smaller than the bin length and buffer
    if (sdiff < inc->info->binlen + SPIKE_THRES && (buf->idx + 1) % buf->size == 0) {
        //Buffer size too small, double it
        buf->data = realloc(buf->data, buf->size*buf->size*sizeof(Spike))
        buf->size = buf->size * buf->size;
        #ifdef DEBUG
        printf("Expanding spike buffer to %d\n"%buf->size);
        #endif
    }

    idx = buf->idx++ % buf->size;
    memcpy(buf->data[idx], spike, sizeof(Spike));

    if (sdiff > inc->info->binlen + SPIKE_THRES) {
        for (i = 0; i < buf->size; i++) {
            idx = (i + buf->idx) % buf->size;
            sptr = &(buf->data[idx]);
            chan = _hash_chan((ushort) s->chan, (ushort) s->unit);
            if (inc->info->chanmap[chan] > 0 && (spike->ts - sptr->ts)
            inc->output[chan, ]
        }
    }

}