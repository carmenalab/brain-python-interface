#include "plexread.h"

ContData* plx_read_continuous(PlexFile* plxfile, ChanType type,
    double start, double stop, int* chans, int nchans) {
    if (!(type == wideband || type == spkc || type == lfp || type == analog))
        return NULL;
    bool alloc_chans = false;
    short buf[MAX_SAMPLES_PER_WAVEFORM];
    size_t headsize = sizeof(PL_DataBlockHeader);
    double first_samp, last_samp, gain;
    double tsfreq = (double) plxfile->header.ADFrequency;
    int chanfreq = plxfile->cont_info[type-wideband][0].ADFreq;
    unsigned long i, f, c, t, stride;
    unsigned long t_off, fedge[2], adj[2], strunc[2] = {0,0};

    FrameSet* frameset = &(plxfile->data[type]);
    DataFrame* frame = &(frameset->frames[0]);
    ContData* data = malloc(sizeof(ContData));
    unsigned long cskip[max((unsigned long) nchans, frame->nblocks)];

    fedge[0] = start < 0 ? 0 : _binary_search(frameset, (TSTYPE) (start*tsfreq));
    fedge[1] = stop < 0 ? frameset->num : _binary_search(frameset, (TSTYPE) (stop*tsfreq))+1;

    first_samp = frameset->frames[fedge[0]].ts / tsfreq * chanfreq;
    last_samp = frameset->frames[fedge[1]-1].ts / tsfreq * chanfreq;
    last_samp += frameset->frames[fedge[1]-1].samples;

    if (start >= 0 && first_samp != start) 
        strunc[0] = ceil(start*chanfreq - first_samp);
    if (stop >= 0 && last_samp != stop)
        strunc[1] = floor(last_samp - stop*chanfreq);

    data->freq = (double) chanfreq;
    data->t_start = (first_samp + strunc[0] - max(0,start)*chanfreq) / data->freq;
    start = start < 0 ? 0 : ceil(first_samp + strunc[0]);
    stop = stop < 0 ? ceil(last_samp) : floor(last_samp - strunc[1]);

#ifdef DEBUG
    printf("Start/stop at (%f, %f), strunc=(%lu, %lu)\n", start, stop, strunc[0], strunc[1]);
#endif
    if (chans == NULL || nchans < 1) {
        chans = malloc(sizeof(int) * frame->nblocks);
        alloc_chans = true;
        for (i = 0; i < frame->nblocks; i++)
            chans[i] = i;
        data->nchans = frame->nblocks;
    } else {
        data->nchans = nchans;
    }
    for (i = 0; i < data->nchans-1; i++)
        cskip[i] = chans[i+1] - chans[i];
    cskip[i] = 1;

    data->len = stop - start;
    data->data = calloc(data->len * data->nchans, sizeof(double));

    for (f = fedge[0]; f < fedge[1]; f++) {
        adj[0] = f == fedge[0] ? strunc[0] : 0;
        adj[1] = f+1 == fedge[1] ? strunc[1] : 0;
        frame = &(plxfile->data[type].frames[f]);
        t_off = frame->ts * chanfreq / tsfreq + adj[0] - start;
        stride = headsize + frame->samples * sizeof(short);
        fseek(plxfile->fp, frame->fpos[0]+chans[0]*stride+headsize, SEEK_SET);

        for (c = 0; c < data->nchans; c++) {
            gain = (double) plxfile->cont_info[type-wideband][chans[c]].Gain;
            fread(buf, sizeof(short), frame->samples, plxfile->fp);
            fseek(plxfile->fp, headsize+stride*(cskip[c]-1), SEEK_CUR);
            for (t = 0; t < frame->samples - adj[1]; t++)
                data->data[(t+t_off)*data->nchans + c] = buf[t+adj[0]] / gain;
        }
    }

    if (alloc_chans) free(chans);
    return data;
}

SpikeData* plx_read_events_spikes(PlexFile* plxfile, ChanType type, double start, double stop, bool waveforms) {
    if (!(type == spike || type == event))
        return NULL;

    short buf[MAX_SAMPLES_PER_WAVEFORM];
    double gain;
    double tsfreq = plxfile->header.ADFrequency;
    unsigned long i, j, k, fedge[2], n = 0;
    FrameSet* frameset = &(plxfile->data[type]);
    DataFrame* frame = frameset->frames;
    SpikeData* spikes = malloc(sizeof(SpikeData));

    fedge[0] = start < 0 ? 0 : _binary_search(frameset, (TSTYPE) start*tsfreq)+1;
    fedge[1] = stop < 0 ? frameset->num : _binary_search(frameset, (TSTYPE) stop*tsfreq);
    spikes->wflen = frame->samples;
    spikes->num = 0;

    for (i = fedge[0]; i < fedge[1]; i++) {
        spikes->num += frameset->frames[i].nblocks;
    }
    if (waveforms)
        spikes->waveforms = malloc(sizeof(double));
    
    spikes->spike = malloc(sizeof(Spike) * spikes->num);
    for (i = fedge[0]; i < fedge[1]; i++) {
        frame = &(frameset->frames[i]);
        fseek(plxfile->fp, frame->fpos[0] + 8, SEEK_SET);
        for (j = 0; j < frame->nblocks; j++) {
            spikes->spike[n].ts = frame->ts / (double) tsfreq;
            fread(buf, 2, sizeof(short), plxfile->fp);
            spikes->spike[n].chan = (int) buf[0];
            spikes->spike[n].unit = (int) buf[1];
            fseek(plxfile->fp, 4, SEEK_CUR);
            
            if (waveforms) {
                gain = (double) plxfile->chan_info[spikes->spike[n].chan].Gain;
                fread(buf, spikes->wflen, sizeof(short), plxfile->fp);
                for (k = 0; k < (unsigned long) spikes->wflen; k++) 
                    spikes->waveforms[n*spikes->wflen + k] = buf[k] / gain;
            } else {
                fseek(plxfile->fp, spikes->wflen*2 + 8, SEEK_CUR);
            }
            n++;
        }
    }
    return spikes;
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