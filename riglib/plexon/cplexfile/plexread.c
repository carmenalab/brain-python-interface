#include "plexread.h"

ContInfo* plx_get_continuous(PlexFile* plxfile, ChanType type, 
    double start, double stop, int* chans, int nchans) {
    if (!(type == wideband || type == spkc || type == lfp || type == analog))
        return NULL;

    if (!plxfile->has_cache)
        return NULL;
    
    double first_samp, last_samp, adj;
    double tsfreq = (double) plxfile->header.ADFrequency;
    int chanfreq = plxfile->cont_info[type-wideband][0].ADFreq;
    unsigned long i;

    FrameSet* frameset = &(plxfile->data[type]);
    DataFrame* frame = &(frameset->frames[0]);
    ContInfo* info = malloc(sizeof(ContInfo));
    long long flen = frameset->num;

    if (frameset->num < 1)
        return NULL;

    info->type = type;
    info->plxfile = plxfile;
    info->start = start;
    info->stop = stop;
    info->_strunc[0] = 0; info->_strunc[1] = 0;

    #ifdef DEBUG
    printf("Looking from %f to %f\n", start, stop);
    #endif

    info->_fedge[0] = start < 0 ? 0 : _binary_search(frameset, (TSTYPE) (start*tsfreq));
    info->_fedge[1] = stop < 0 ? flen : _binary_search(frameset, (TSTYPE) (stop*tsfreq))+1;

    first_samp = frameset->frames[info->_fedge[0]].ts / tsfreq * chanfreq;
    last_samp = frameset->frames[info->_fedge[1]-1].ts / tsfreq * chanfreq;
    last_samp += frameset->frames[info->_fedge[1]-1].samples;
    start = max(0, start)*chanfreq;
    stop = min(last_samp/chanfreq, stop)*chanfreq;

    if (first_samp != start && first_samp < start)
        info->_strunc[0] = ceil(start - first_samp);
    if (last_samp != stop && last_samp > stop) {
        info->_strunc[1] = floor(last_samp - stop);
    }

    info->freq = (double) chanfreq;
    adj = first_samp + info->_strunc[0] - start;
    info->t_start = (adj - floor(adj)) / info->freq;
    info->_start = start;

    if (nchans < 1) {
        info->chans = malloc(sizeof(int) * frame->nblocks);
        for (i = 0; i < frame->nblocks; i++)
            info->chans[i] = i;
        info->nchans = frame->nblocks;
    } else {
        info->chans = memcpy(malloc(sizeof(int) * nchans), chans, sizeof(int)*nchans);
        info->nchans = nchans;
    }
    info->cskip = malloc(info->nchans*sizeof(int));
    for (i = 0; i < info->nchans-1; i++)
        info->cskip[i] = info->chans[i+1] - info->chans[i];
    info->cskip[i] = 1;

    #ifdef DEBUG
    printf("Continuous slice:\n");
    printf("\ttsfreq=%f, chanfreq=%d\n", tsfreq, chanfreq);
    printf("\tsamples=[%f:%f], trunc=[%lu, %lu]\n", 
        first_samp, last_samp, info->_strunc[0], info->_strunc[1]);
    printf("\tstart=[%f:%f], t_start=%f\n", start, stop, info->t_start);
    printf("\tfedge=[%llu:%llu]\n", info->_fedge[0], info->_fedge[1]);
    #endif

    info->len = ceil(stop - start - info->t_start*chanfreq);
    return info;
}

int plx_read_continuous(ContInfo* info, double* data) {
    double gain;
    unsigned long t_off, adj[2];
    unsigned long c, t, stride;
    long long f;
    short buf[MAX_SAMPLES_PER_WAVEFORM];
    size_t headsize = sizeof(PL_DataBlockHeader), readsize;
    double tsfreq = (double) info->plxfile->header.ADFrequency;
    DataFrame* frame = &(info->plxfile->data[info->type].frames[0]);
    PL_SlowChannelHeader* chan_info = info->plxfile->cont_info[info->type - wideband];
    int chanfreq = chan_info[0].ADFreq;

    #ifdef DEBUG
    printf("Channels:");
    unsigned long i;
    for (i = 0; i < info->nchans; i++)
        printf("\t\t%d -- %d\n", info->chans[i], info->cskip[i]);
    #endif

    for (f = info->_fedge[0]; f < info->_fedge[1]; f++) {
        adj[0] = f == info->_fedge[0] ? info->_strunc[0] : 0;
        adj[1] = f+1 == info->_fedge[1] ? info->_strunc[1] : 0;

        frame = &(info->plxfile->data[info->type].frames[f]);
        t_off = frame->ts * chanfreq / tsfreq - info->_start + adj[0];
        stride = headsize + frame->samples * sizeof(short);
        fseek(info->plxfile->fp, frame->fpos+info->chans[0]*stride+headsize, SEEK_SET);

        for (c = 0; c < info->nchans; c++) {
            gain = (double) chan_info[info->chans[c]].Gain;
            readsize = fread(buf, sizeof(short), frame->samples, info->plxfile->fp);
            if (readsize != frame->samples)
                return -1;
            fseek(info->plxfile->fp, headsize+stride*(info->cskip[c]-1), SEEK_CUR);
            for (t = 0; t < frame->samples - adj[1]; t++)
                data[(t+t_off)*info->nchans + c] = buf[t+adj[0]] / gain;
        }
    }

    return 0;
}

void free_continfo(ContInfo* info) {
    free(info->cskip);
    free(info->chans);
    free(info);
}


SpikeInfo* plx_get_discrete(PlexFile* plxfile, ChanType type, double start, double stop) {
    unsigned long i;

    if (!(type == spike || type == event)) {
        fprintf(stderr, "Invalid channel type\n");
        return NULL;
    }

    if (!plxfile->has_cache) {
        fprintf(stderr, "Plexfile has not been loaded\n");
        return NULL;
    }

    FrameSet* frameset = &(plxfile->data[type]);
    if (frameset->num < 1) {
        fprintf(stderr, "Plexfile contains no frames of this channel type\n");
        return NULL;
    }

    SpikeInfo* info = malloc(sizeof(SpikeInfo));
    DataFrame* frame = frameset->frames;

    double tsfreq = plxfile->header.ADFrequency;
    double final = plxfile->header.LastTimestamp;
    long long flen = (long long) frameset->num;
    TSTYPE fstart = (TSTYPE) (start < 0 ? 0 : floor(start*tsfreq + .5));
    TSTYPE fstop  = (TSTYPE) (stop < 0 ? final : floor(stop*tsfreq + .5));

    long long lower = _binary_search(frameset, fstart)-SPIKE_SEARCH;
    long long upper = _binary_search(frameset, fstop )+SPIKE_SEARCH;

    #ifdef DEBUG
    printf("Searching from frame %lld to %lld: timestamp %lu-%lu\n", lower, upper, fstart, fstop);
    #endif

    info->plxfile = plxfile;
    info->start = start;
    info->stop = stop;
    info->type = type;
    info->_fedge[0] = start < 0 || lower < 0 ? 0 : lower;
    info->_fedge[1] = stop < 0 || upper > flen ? flen : upper; 
    info->wflen = frameset->num > 0 ? frame->samples : 0;
    info->num = 0;

    for (i = info->_fedge[0]; i < info->_fedge[1]; i++) {
        frame = &(frameset->frames[i]);
        if (fstart <= frame->ts && frame->ts < fstop)
            info->num += frameset->frames[i].nblocks;
    }
    return info;
}

int plx_read_discrete(SpikeInfo* info, Spike* data) {
    short buf[2];
    unsigned long i, j, n = 0;

    FrameSet* frameset = &(info->plxfile->data[info->type]);
    DataFrame* frame = frameset->frames;

    double tsfreq = info->plxfile->header.ADFrequency;
    double final = info->plxfile->header.LastTimestamp;
    TSTYPE fstart = (TSTYPE) (info->start < 0 ? 0 : floor(info->start*tsfreq + .5));
    TSTYPE fstop  = (TSTYPE) (info->stop < 0 ? final : floor(info->stop*tsfreq + .5));

    for (i = info->_fedge[0]; i < info->_fedge[1]; i++) {
        frame = &(frameset->frames[i]);
        fseek(info->plxfile->fp, frame->fpos + 8, SEEK_SET);
        for (j = 0; j < frame->nblocks; j++) {
            if (fstart <= frame->ts && frame->ts < fstop) {
                if (fread(buf, 2, sizeof(short), info->plxfile->fp) != 2)
                    return -1;
                data[n].ts = frame->ts / (double) tsfreq;
                data[n].chan = (int) buf[0];
                data[n].unit = (int) buf[1];
                fseek(info->plxfile->fp, 4 + info->wflen*2 + 8, SEEK_CUR);
                n++;
            }
        }
    }
    return 0;
}

int plx_read_waveforms(SpikeInfo* info, double* data) {
    double gain;
    unsigned long i, j, k, n = 0;
    short buf[MAX_SAMPLES_PER_WAVEFORM], chan;

    FrameSet* frameset = &(info->plxfile->data[info->type]);
    DataFrame* frame = frameset->frames;

    double tsfreq = info->plxfile->header.ADFrequency;
    double final = info->plxfile->header.LastTimestamp;
    TSTYPE fstart = (TSTYPE) (info->start < 0 ? 0 : floor(info->start*tsfreq + .5));
    TSTYPE fstop  = (TSTYPE) (info->stop < 0 ? final : floor(info->stop*tsfreq + .5));
    size_t readsize;

    for (i = info->_fedge[0]; i < info->_fedge[1]; i++) {
        frame = &(frameset->frames[i]);
        fseek(info->plxfile->fp, frame->fpos + 8, SEEK_SET);
        for (j = 0; j < frame->nblocks; j++) {
            if (fstart <= frame->ts && frame->ts < fstop) {
                if (fread(&chan, sizeof(short), 1, info->plxfile->fp) != 1)
                    return -1;
                gain = (double) info->plxfile->chan_info[chan-1].Gain;
                fseek(info->plxfile->fp, 6, SEEK_CUR);
                readsize = fread(buf, sizeof(short), info->wflen, info->plxfile->fp);
                if (readsize != (unsigned long) info->wflen)
                    return -1;
                fseek(info->plxfile->fp, 8, SEEK_CUR);
                for (k = 0; k < (unsigned long) info->wflen; k++) 
                    data[n*info->wflen + k] = buf[k] / gain;
                //if (chan == 256)
                //    printf("buf=%d,%d,%d,...,%d / %f\n", buf[0], buf[1], buf[2], buf[info->wflen-1], gain);
                n++;
            }
        }
    }
    return 0;
}

void free_spikeinfo(SpikeInfo* info) {
    free(info);
}


IterSpike* plx_iterate_discrete(SpikeInfo* info) {
    IterSpike* iter = calloc(1, sizeof(IterSpike));
    if (iter == NULL)
        return NULL;

    iter->plxfile = info->plxfile;
    iter->info = info;

    double final = info->plxfile->header.LastTimestamp;
    double tsfreq = info->plxfile->header.ADFrequency;

    iter->fstart = (TSTYPE) (iter->info->start < 0 ? 0 : floor(iter->info->start*tsfreq + .5));
    iter->fstop  = (TSTYPE) (iter->info->stop < 0 ? final : floor(iter->info->stop*tsfreq + .5));
    iter->i = iter->info->_fedge[0];

    return iter;
}

int plx_iterate(IterSpike* iter, Spike* data) {
    FrameSet* frameset = &(iter->plxfile->data[iter->info->type]);
    DataFrame* frame = frameset->frames + iter->i;
    double lasttime;
    short buf[2];
    size_t stride = 16 + iter->info->wflen*2;
    fseek(iter->plxfile->fp, frame->fpos + stride*iter->j + 8, SEEK_SET);
    while (iter->i < iter->info->_fedge[1]) {
        if (fread(buf, 2, sizeof(short), iter->plxfile->fp) != 2) {
            fprintf(stderr, "Error reading plx file\n");
            return -1;
        }

        data->ts = frame->ts / (double) iter->info->plxfile->header.ADFrequency;
        data->chan = (int) buf[0];
        data->unit = (int) buf[1];
        lasttime = frame->ts;
        fseek(iter->plxfile->fp, 4 + iter->info->wflen*2 + 8, SEEK_CUR);

        if (++iter->j >= frame->nblocks) {
            iter->j = 0;
            iter->i++;
            frame = frameset->frames + iter->i;
            fseek(iter->plxfile->fp, frame->fpos + 8, SEEK_SET);
        }

        if (iter->fstart <= lasttime && lasttime < iter->fstop) {
            #ifdef DEBUG2
            printf("Found spike %d, %d (%d, %d) at %f\n", data->chan, data->unit, iter->i, iter->j, data->ts);
            #endif
            return 0;
        }
    }

    return 1;
}

void free_iterspike(IterSpike* iter) {
    free(iter);
}


long long _binary_search(FrameSet* frameset, TSTYPE ts) {
    long long mid = (long long) frameset->num / 2, 
              left = 0, 
              right = (long long) frameset->num;

    DataFrame* frame = &(frameset->frames[mid]);

    while (right - left > 1) {
        frame = &(frameset->frames[mid]);
        //printf("%lu, %lu, %lu: %llu\n", left, mid, right, frame->ts);
        if (ts < frame->ts) {
            right = mid;
        } else if (ts > frame->ts) {
            left = mid;
        } else if (ts == frame->ts)
            return mid;
        mid = (right - left) / 2 + left;
    }

    return mid;
}

