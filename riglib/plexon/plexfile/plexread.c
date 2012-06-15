#include "plexread.h"

ContData* plx_read_continuous(PlexFile* plxfile, ChanType type,
    double start, double stop, int* chans, int nchans) {
    if (!(type == wideband || type == spkc || type == lfp || type == analog))
        return NULL;
    int* cskip;
    short buf[MAX_SAMPLES_PER_WAVEFORM];
    double first_ts, last_ts, gain;
    double tsfreq = (double) plxfile->header.ADFrequency;
    int chanfreq = plxfile->cont_info[type-wideband][0].ADFreq;
    unsigned long i, f, c, t, stride, adj[2];
    unsigned long t_off, fedge[2], strunc[2] = {0,0};

    FrameSet* frameset = &(plxfile->data[type]);
    DataFrame* frame = &(frameset->frames[0]);
    ContData* data = malloc(sizeof(ContData));

    fedge[0] = start < 0 ? 0 : _binary_search(frameset, (TSTYPE) (start*chanfreq));
    fedge[1] = stop < 0 ? frameset->num : _binary_search(frameset, (TSTYPE) (stop*chanfreq))+1;

    first_ts = frameset->frames[fedge[0]].ts;
    last_ts = frameset->frames[fedge[1]-1].ts + frameset->frames[fedge[1]-1].samples;
    if (first_ts != start) 
        strunc[0] = ceil(start*chanfreq) - first_ts;
    if (last_ts != stop)
        strunc[1] = last_ts - floor(stop*chanfreq);
    data->t_start = ((first_ts+strunc[0]) - start*chanfreq) / tsfreq;
    data->freq = (double) chanfreq;
    start = start < 0 ? 0 : first_ts + strunc[0];
    stop = stop < 0 ? last_ts : last_ts - strunc[1];
    printf("Found edges at (%lu, %lu), strunc=(%lu, %lu)\n", fedge[0], fedge[1], strunc[0], strunc[1]);
    if (chans == NULL || nchans < 1) {
        chans = malloc(sizeof(int) * frame->nblocks);
        cskip = malloc(sizeof(int) * frame->nblocks);
        for (i = 0; i < frame->nblocks; i++) {
            chans[i] = i;
            cskip[i] = 1;
        }
        data->nchans = frame->nblocks;
    } else {
        //Compute the channel differential
        cskip = malloc(sizeof(int) * (nchans + 1));
        for (i = 0; i < (unsigned long) nchans-1; i++)
            cskip[i] = chans[i+1] - chans[i];
        cskip[nchans] = frame->nblocks - chans[i];
        data->nchans = nchans;
    }

    data->len = stop - start;
    data->data = calloc(data->len * data->nchans, sizeof(double));

    for (f = fedge[0]; f < fedge[1]; f++) {
        adj[0] = f == fedge[0] ? strunc[0] : 0;
        adj[1] = f-1 == fedge[1] ? strunc[1] : 0;
        frame = &(plxfile->data[type].frames[f]);

        t_off = frame->ts + adj[0] - start;
        stride = frame->samples * sizeof(short) + sizeof(PL_DataBlockHeader);
        fseek(plxfile->fp, frame->fpos[0]+sizeof(PL_DataBlockHeader), SEEK_SET);

        for (c = 0; c < data->nchans; c++) {
            gain = (double) plxfile->cont_info[type-wideband][chans[i]].Gain;
            fread(buf, sizeof(short), frame->samples, plxfile->fp);
            fseek(plxfile->fp, stride*cskip[c], SEEK_CUR);
            for (t = 0; t < frame->samples - adj[1]; t++)
                data->data[(t+t_off)*nchans + c] = buf[t+adj[0]] / gain;
        }
    }

    free(cskip);
    if (data->nchans == frame->nblocks)
        free(chans);

    return data;
}

SpikeData* plx_read_event_spike(PlexFile* plxfile, ChanType type, double start, double stop, bool waveforms) {
    if (!(type == spike || type == event))
        return NULL;

    int tsfreq = plxfile->header.ADFrequency;
    short buf[MAX_SAMPLES_PER_WAVEFORM];
    double changain;
    unsigned long i, j, k, fedge[2], n = 0;
    DataFrame* frame;
    SpikeData* spikes = malloc(sizeof(SpikeData));

    fedge[0] = start < 0 ? 0 : _binary_search(&(plxfile->data[type]), start);
    fedge[1] = stop < 0 ? plxfile->data[type].num : _binary_search(&(plxfile->data[type]), stop);
    spikes->wflen = plxfile->data[type].frames[0].samples;
    spikes->num = 0;

    for (i = 0; i < plxfile->data[type].num; i++) {
        spikes->num += plxfile->data[type].frames[i].nblocks;
    }
    if (waveforms)
        spikes->waveforms = malloc(sizeof(double));
    
    spikes->spike = malloc(sizeof(Spike) * spikes->num);
    for (i = fedge[0]; i < fedge[1]; i++) {
        frame = &(plxfile->data[type].frames[i]);
        fseek(plxfile->fp, frame->fpos[0] + 8, SEEK_SET);
        for (j = 0; j < frame->nblocks; j++) {
            spikes->spike[n].ts = frame->ts / (double) tsfreq;
            fread(buf, 2, sizeof(short), plxfile->fp);
            spikes->spike[n].chan = (int) buf[0];
            spikes->spike[n].unit = (int) buf[1];
            fseek(plxfile->fp, 4, SEEK_CUR);
            
            if (waveforms) {
                changain = (double) plxfile->chan_info[spikes->spike[n].chan].Gain;
                fread(buf, spikes->wflen, sizeof(short), plxfile->fp);
                for (k = 0; k < (unsigned long) spikes->wflen; k++) 
                    spikes->waveforms[n*spikes->wflen + k] = buf[k] / changain;
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