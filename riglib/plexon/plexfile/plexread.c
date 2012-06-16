#include "plexread.h"

ContInfo* plx_get_continuous(PlexFile* plxfile, ChanType type, 
    double start, double stop, int* chans, int nchans) {
    if (!(type == wideband || type == spkc || type == lfp || type == analog))
        return NULL;
    
    double first_samp, last_samp;
    double tsfreq = (double) plxfile->header.ADFrequency;
    int chanfreq = plxfile->cont_info[type-wideband][0].ADFreq;
    unsigned long i;

    FrameSet* frameset = &(plxfile->data[type]);
    DataFrame* frame = &(frameset->frames[0]);
    ContInfo* info = malloc(sizeof(ContInfo));

    info->type = type;
    info->plxfile = plxfile;
    info->_strunc[0] = 0; info->_strunc[1] = 0;
    info->_fedge[0] = start < 0 ? 0 : _binary_search(frameset, (TSTYPE) (start*tsfreq));
    info->_fedge[1] = stop < 0 ? frameset->num : _binary_search(frameset, (TSTYPE) (stop*tsfreq))+1;

    first_samp = frameset->frames[info->_fedge[0]].ts / tsfreq * chanfreq;
    last_samp = frameset->frames[info->_fedge[1]-1].ts / tsfreq * chanfreq;
    last_samp += frameset->frames[info->_fedge[1]-1].samples;

    if (start >= 0 && first_samp != start) 
        info->_strunc[0] = ceil(start*chanfreq - first_samp);
    if (stop >= 0 && last_samp != stop)
        info->_strunc[1] = floor(last_samp - stop*chanfreq);

    info->t_start = (first_samp + info->_strunc[0] - max(0,start)*chanfreq) / info->freq;
    start = start < 0 ? 0 : ceil(first_samp + info->_strunc[0]);
    stop = stop < 0 ? ceil(last_samp) : floor(last_samp - info->_strunc[1]);
    info->_start = start;
    info->start = start / info->freq;
    info->stop = stop / info->freq;
    info->freq = (double) chanfreq;

    if (chans == NULL || nchans < 1) {
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

    info->len = ceil(stop - start);
    return info;
}

void plx_read_continuous(ContInfo* info, double* data) {
    double gain;
    unsigned long t_off, adj[2];
    unsigned long int f, c, t, stride;
    short buf[MAX_SAMPLES_PER_WAVEFORM];
    size_t headsize = sizeof(PL_DataBlockHeader), readsize;
    double tsfreq = (double) info->plxfile->header.ADFrequency;
    DataFrame* frame = &(info->plxfile->data[info->type].frames[0]);
    PL_SlowChannelHeader* chan_info = info->plxfile->cont_info[info->type - wideband];
    int chanfreq = chan_info[0].ADFreq;

    for (f = info->_fedge[0]; f < info->_fedge[1]; f++) {
        adj[0] = f == info->_fedge[0] ? info->_strunc[0] : 0;
        adj[1] = f+1 == info->_fedge[1] ? info->_strunc[1] : 0;

        frame = &(info->plxfile->data[info->type].frames[f]);
        t_off = frame->ts * chanfreq / tsfreq + adj[0] - info->_start;
        stride = headsize + frame->samples * sizeof(short);
        fseek(info->plxfile->fp, frame->fpos[0]+info->chans[0]*stride+headsize, SEEK_SET);

        for (c = 0; c < info->nchans; c++) {
            gain = (double) chan_info[info->chans[c]].Gain;
            readsize = fread(buf, sizeof(short), frame->samples, info->plxfile->fp);
            assert(readsize == frame->samples);
            fseek(info->plxfile->fp, headsize+stride*(info->cskip[c]-1), SEEK_CUR);
            for (t = 0; t < frame->samples - adj[1]; t++)
                data[(t+t_off)*info->nchans + c] = buf[t+adj[0]] / gain;
        }
    }
}

SpikeInfo* plx_get_discrete(PlexFile* plxfile, ChanType type, double start, double stop) {
    if (!(type == spike || type == event))
        return NULL;

    unsigned long i;
    double tsfreq = plxfile->header.ADFrequency;
    FrameSet* frameset = &(plxfile->data[type]);
    SpikeInfo* info = malloc(sizeof(SpikeInfo));
    DataFrame* frame = frameset->frames;

    info->plxfile = plxfile;
    info->start = start;
    info->stop = stop;
    info->type = type;
    info->_fedge[0] = start < 0 ? 0 : _binary_search(frameset, (TSTYPE) start*tsfreq)+1;
    info->_fedge[1] = stop < 0 ? frameset->num : _binary_search(frameset, (TSTYPE) stop*tsfreq);
    info->wflen = frameset->num > 0 ? frame->samples : 0;
    info->num = 0;

    for (i = info->_fedge[0]; i < info->_fedge[1]; i++) {
        info->num += frameset->frames[i].nblocks;
    }
    return info;
}

void plx_read_discrete(SpikeInfo* info, Spike* data) {
    short buf[2];
    size_t readsize;
    unsigned long i, j, n = 0;
    double tsfreq = info->plxfile->header.ADFrequency;
    FrameSet* frameset = &(info->plxfile->data[info->type]);
    DataFrame* frame = frameset->frames;

    for (i = info->_fedge[0]; i < info->_fedge[1]; i++) {
        frame = &(frameset->frames[i]);
        fseek(info->plxfile->fp, frame->fpos[0] + 8, SEEK_SET);
        for (j = 0; j < frame->nblocks; j++) {
            readsize = fread(buf, 2, sizeof(short), info->plxfile->fp);
            assert(readsize == 2);
            data[n].ts = frame->ts / (double) tsfreq;
            data[n].chan = (int) buf[0];
            data[n].unit = (int) buf[1];
            fseek(info->plxfile->fp, 4 + info->wflen*2 + 8, SEEK_CUR);
            n++;
        }
    }
}

void plx_read_waveforms(SpikeInfo* info, double* data) {
    double gain;
    size_t readsize;
    unsigned long i, j, k, n = 0;
    short buf[MAX_SAMPLES_PER_WAVEFORM];
    FrameSet* frameset = &(info->plxfile->data[info->type]);
    DataFrame* frame = frameset->frames;

    for (i = info->_fedge[0]; i < info->_fedge[1]; i++) {
        frame = &(frameset->frames[i]);
        fseek(info->plxfile->fp, frame->fpos[0] + 8, SEEK_SET);
        for (j = 0; j < frame->nblocks; j++) {
            readsize = fread(buf, 1, sizeof(short), info->plxfile->fp);
            assert(readsize == 1);
            gain = (double) info->plxfile->chan_info[buf[0]].Gain;
            fseek(info->plxfile->fp, 6, SEEK_CUR);
            readsize = fread(buf, info->wflen, sizeof(short), info->plxfile->fp);
            assert(readsize == (unsigned long) info->wflen);
            fseek(info->plxfile->fp, 8, SEEK_CUR);
            for (k = 0; k < (unsigned long) info->wflen; k++) 
                data[n*info->wflen + k] = buf[k] / gain;
        }
    }
}

unsigned long _binary_search(FrameSet* frameset, TSTYPE ts) {
    unsigned long mid = frameset->num / 2, left = 0, right = frameset->num;
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

void free_continfo(ContInfo* info) {
    free(info->cskip);
    free(info->chans);
    free(info);
}
void free_spikeinfo(SpikeInfo* info) {
    free(info);
}