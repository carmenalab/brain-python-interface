#include "psth.h"

uint _hash_chan(ushort chan, ushort unit) {
    uint ichan = (uint) chan, iunit = (uint) unit;
    return (((ichan & 1023) << 3) | (iunit & 7));
}

double boxcar(double start, double ts, double* params) {
    return 1;
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

extern int bin_inc_spike(BinInc* inc, Spike* spike, double* output) {
    uint i, j, chan, num;
    unsigned long idx, min;
    double binlen, sdiff, curtime, val;
    SpikeBuf* buf = &(inc->spikes);
    Spike* sptr, *oldest;

    oldest = buf->data + (buf->last % buf->size);
    sdiff = buf->idx == 0 ? 0 : spike->ts - oldest->ts;
    binlen = inc->info->binlen + SPIKE_FUZZ;

    //If time range in buffer is smaller than the bin length
    if (sdiff < binlen && buf->idx >= buf->size) {
        //Buffer size too small, double it
        sptr = calloc(2*buf->size, sizeof(Spike));
        i = buf->idx % buf->size;
        num = buf->size - i;
        memcpy(sptr, (void*) buf->data+i, sizeof(Spike)*num);
        memcpy(sptr+num, (void*) buf->data, sizeof(Spike)*i);
        free(buf->data);
        buf->last = 0;
        buf->idx = buf->size;
        buf->size *= 2;
        buf->data = sptr;
        #ifdef DEBUG
        //printf("Copied %d spikes from %d, idx=%lu->%lu\n", num, i, idx, buf->idx);
        printf("Expanding spike buffer to %d, sdiff=%f < %f\n", buf->size, sdiff, binlen);
        #endif
    }

    idx = buf->idx++ % buf->size;
    memcpy(buf->data + idx, (void*) spike, sizeof(Spike));
    if (buf->idx - buf->last > buf->size)
        buf->last++;

    curtime = inc->times[inc->_tidx];
    //We've exceeded the current bin time, count up all the spikes
    if ( spike->ts > curtime + SPIKE_FUZZ) {
        j = 0;
        min = buf->idx < buf->size ? buf->idx : buf->size;
        for (i = 0; i < min; i++) {
            idx = (buf->idx + buf->size - i - 1) % buf->size;
            sptr = buf->data + idx;
            sdiff = curtime - sptr->ts;
            if (sdiff > binlen) {
                #ifdef DEBUG
                printf("Stopping early at %d, %f > %f\n", (buf->idx - i - 1), sdiff, binlen);
                #endif
                break;
            }

            if ( FLT_EPSILON < sdiff && (sdiff - inc->info->binlen < FLT_EPSILON || 
                 fabs(sdiff - inc->info->binlen) < FLT_EPSILON) ) {
                chan = _hash_chan((ushort) sptr->chan, (ushort) sptr->unit);
                if (inc->info->chanmap[chan] > 0) {
                    //printf("Add (%d, %d), sdiff=%f for %f into %d\n", sptr->chan, sptr->unit, sptr->ts, curtime, inc->info->chanmap[chan] - 1);
                    val = (*(inc->info->binfunc))(curtime, sptr->ts, inc->info->params);
                    output[inc->info->chanmap[chan] - 1] += val;
                    j++;
                }
            } 
            #ifdef DEBUG
            else {
                printf("Exclude (%d, %d), sdiff=%f for %f, diff=%f\n", sptr->chan, sptr->unit, sptr->ts, curtime, sdiff - inc->info->binlen);
            }
            #endif
        }
        #ifdef DEBUG
        printf("filled bin %d with %d spikes\n", inc->_tidx, j);
        #endif
        if (++inc->_tidx > inc->tlen)
            return 2;

        return 1;
    }

    return 0;
}

extern void free_bininfo(BinInfo* info) {
    free(info);
}

extern void free_bininc(BinInc* inc) {
    free(inc->times);
    free(inc);
}