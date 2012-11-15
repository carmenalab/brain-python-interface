#include "psth.h"

uint _hash_chan(ushort chan, ushort unit) {
    uint ichan = (uint) chan, iunit = (uint) unit;
    return (((ichan & 1023) << 3) | (iunit & 7));
}

double boxcar(double start, double ts, double* params) {
    //First parameter is for box width
    return (double) (start - ts < params[0]);
}

double gaussian(double start, double ts, double* params) {
    //First parameter is for box width
    return (double) (start - ts < params[0]);
}

extern BinInfo* bin_init(int* chans, size_t clen, double binlen, char* funcname, double* params) {
    int i, idx;
    uint nparams = 0;
    Channel* channels = (Channel*) chans;
    BinInfo* info = (BinInfo*) calloc(1, sizeof(BinInfo));
    info->binlen = binlen;
    info->nunits = clen;

    for (i = 0; i < info->nunits; i++) {
        idx = _hash_chan((ushort) channels[i].chan, (ushort) channels[i].unit);
        info->chanmap[idx] = i+1;
        #ifdef DEBUG
        printf("Setting hash(%d, %d) = %d to %d\n", channels[i].chan, channels[i].unit, idx, i+1);
        #endif
    }

    if (strcmp(funcname, "boxcar") == 0) {
        #ifdef DEBUG
        printf("Using boxcar filter, binlen=%f\n", binlen);
        #endif
        info->binfunc = &boxcar;
        nparams = 0;
        info->params[0] = binlen;
    } else if (strcmp(funcname, "gaussian") == 0) {
        #ifdef DEBUG
        printf("Using gaussian filter, mean=%f, std=%f\n", params[0], params[1]);
        #endif
        info->binfunc = &gaussian;
        nparams = 2;
    } else {
        return NULL;
    }

    if (params != NULL)
        memcpy(info->params, params, nparams);

    return info;
}

extern void bin_spikes(BinInfo* info, Spike* spikes, uint nspikes, double* output) {
    uint i, idx;
    double sdiff, val, curtime = spikes[nspikes-1].ts;

    //Since spike times are only vaguely ordered, find the largest timestamp
    for (i = nspikes-1; i > nspikes-100 && i > 0; i--) {
        if (spikes[i].ts > curtime)
            curtime = spikes[i].ts;
    }

    //Search through spikes in reverse order
    for (i = nspikes-1; i > 0; i--) {
        idx = _hash_chan((ushort) spikes[i].chan, (ushort) spikes[i].unit);
        sdiff = curtime - spikes[i].ts;
        //Break when the time is greater than spike fuzz
        if (sdiff >= info->binlen + SPIKE_FUZZ) {
            return;
        }

        //If the unit is in list and it's within binlen, count
        if (info->chanmap[idx] > 0 && 0 <= sdiff && sdiff < info->binlen) {
            val = (*(info->binfunc))(curtime, spikes[i].ts, info->params);
            output[info->chanmap[idx] - 1] += val;
            #ifdef DEBUG
            printf("Adding %f to (%d, %d)\n", val, spikes[i].chan, spikes[i].unit);
            #endif
        }
    }

    #ifdef DEBUG
    printf("End of spikes -- not enough for full counting\n");
    #endif
}

extern BinInc* bin_incremental(BinInfo* info, double* times, uint tlen) {
    BinInc* inc = (BinInc*) calloc(1, sizeof(BinInc));
    inc->info = info;
    inc->spikes.data = calloc(BUF_INIT_SIZE, sizeof(Spike));
    inc->spikes.size = BUF_INIT_SIZE;
    inc->times = malloc(sizeof(double)*tlen);
    memcpy(inc->times, times, tlen*sizeof(double));
    inc->tlen = tlen;

    return inc;
}

extern bool bin_inc_spike(BinInc* inc, Spike* spike, double* output) {
    uint i, idx, chan;
    double binlen, sdiff, curtime, val;
    SpikeBuf* buf = &(inc->spikes);
    Spike* sptr, *last;

    last = &(buf->data[(buf->idx+1)%buf->size]);
    if (buf->idx < buf->size)
        last = &(buf->data[0]);

    sdiff = buf->idx == 0 ? 0 : spike->ts - last->ts;
    binlen = inc->info->binlen + SPIKE_FUZZ;

    //If time range in buffer is smaller than the bin length
    if (sdiff < binlen && (buf->idx + 1) % buf->size == 0) {
        //Buffer size too small, double it
        buf->data = realloc(buf->data, buf->size*buf->size*sizeof(Spike));
        buf->size = buf->size * buf->size;
        #ifdef DEBUG
        printf("Expanding spike buffer to %d\n", buf->size);
        #endif
    }

    idx = buf->idx++ % buf->size;
    memcpy(&(buf->data[idx]), (void*) spike, sizeof(Spike));

    //We've exceeded the current bin time, count up all the spikes
    curtime = inc->times[inc->_tidx];
    if (curtime - last->ts >= binlen) {
        for (i = 0; i < buf->size; i++) {
            idx = (buf->idx - i - 1) % buf->size;
            sptr = &(buf->data[idx]);
            sdiff = curtime - sptr->ts;
            chan = _hash_chan((ushort) sptr->chan, (ushort) sptr->unit);
            if (inc->info->chanmap[chan] > 0 && 0 <= sdiff && sdiff < inc->info->binlen) {
                val = (*(inc->info->binfunc))(curtime, sptr->ts, inc->info->params);
                output[inc->info->chanmap[chan] - 1] += val;
            }
        }
        inc->_tidx ++;
        return true;
    }

    return false;
}

extern void free_bininfo(BinInfo* info) {
    free(info);
}

extern void free_bininc(BinInc* inc) {
    free(inc->times);
    free(inc);
}