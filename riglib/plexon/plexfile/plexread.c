#include "plexread.h"

ContData* plx_read_continuous(PlexFile* plxfile, ChanType type,
    double start, double stop, int* chans, int nchans) {
    if (!(type == wideband || type == spkc || type == lfp || type == analog))
        return NULL;
    int i, f, c, t, stride;
    short buf[MAX_SAMPLES_PER_WAVEFORM];
    double tsfreq = (double) plxfile->header.ADFrequency;
    double chanfreq = (double) plxfile->cont_info[type-wideband][0].ADFreq;
    double gain = (double) plxfile->cont_info[type-wideband][0].Gain;
    unsigned long t_off, fedge[2], strunc[2] = {0,0};

    FrameSet* frameset = &(plxfile->data[type]);
    DataFrame* frame = &(frameset->frames[0]);
    ContData* data = malloc(sizeof(ContData));

    data->freq = chanfreq;
    data->t_start = frame->ts / tsfreq;

    fedge[0] = start < 0 ? start : _binary_search(frameset, (TSTYPE) start*chanfreq);
    fedge[1] = stop < 0 ? frameset->num-1 : _binary_search(frameset, (TSTYPE) stop*chanfreq);
    if (frameset->frames[fedge[0]].ts / tsfreq != start)
        strunc[0] = 

    if (chans == NULL || nchans < 1) {
        chans = malloc(sizeof(int) * frame->nblocks);
        for (i = 0; i < frame->nblocks; i++)
            chans[i] = 1;
        data->nchans = frame->nblocks;
    } else {
        //Compute the channel differential
        for (i = 0; i < nchans-1; i++) {
            chans[i] = chans[i+1] - chans[i];
        }
        data->nchans = nchans;
    }

    data->len = (frameset->frames[fedge[-1]].ts - frameset->frames[fedge[0]].ts) / tsfreq * data->freq;
    data->len += frameset->frames[fedge[-1]].samples - strunc[0] - strunc[1];
    data->data = calloc(data->len * data->nchans, sizeof(double));

    for (f = start; f < stop; f++) {
        frame = &(plxfile->data[type].frames[f]);
        t_off = (frame->ts / tsfreq - data->t_start) * chanfreq;
        stride = frame->samples * sizeof(short) + sizeof(PL_DataBlockHeader);
        fseek(plxfile->fp, frame->fpos[0]+sizeof(PL_DataBlockHeader), SEEK_SET);

        for (c = 0; c < data->nchans; c++) {
            fread(buf, sizeof(short), frame->samples, plxfile->fp);
            fseek(plxfile->fp, stride*chans[c], SEEK_CUR);
            for (t = 0; t < frame->samples; t++)
                data->data[(t+t_off)*nchans + c] = buf[t] / gain;
        }
    }

    if (nchans == frame->nblocks)
        free(chans);

    return data;
}
/*
SpikeData* plx_readall_spikes(PlexFile* plxfile, bool waveforms) {
    int i, j, k, n = 0;
    int tsfreq = plxfile->header.ADFrequency;
    short buf[MAX_SAMPLES_PER_WAVEFORM];
    double changain;
    DataFrame* frame;
    SpikeData* spikes = malloc(sizeof(SpikeData));
    spikes->wflen = plxfile->spikes.frames[0].samples;
    spikes->num = 0;
    for (i = 0; i < plxfile->spikes.num; i++) {
        spikes->num += plxfile->spikes.frames[i].nblocks;
    }
    if (waveforms)
        spikes->waveforms = malloc(sizeof(double));
    
    spikes->spike = malloc(sizeof(Spike) * spikes->num);
    for (i = 0; i < plxfile->spikes.num; i++) {
        frame = &(plxfile->spikes.frames[i]);
        fseek(plxfile->fp, frame->fpos[0] + 8, SEEK_SET);
        for (j = 0; j < frame->nblocks; j++) {
            spikes->spike[n].ts = frame->ts / (double) tsfreq;
            fread(buf, 2, sizeof(short), plxfile->fp);
            spikes->spike[n].chan = (int) buf[0];
            spikes->spike[n].unit = (int) buf[1];
            fseek(plxfile->fp, 4, SEEK_CUR);
            
            changain = (double) plxfile->chan_info[spikes->spike[n].chan].Gain;
            if (waveforms) {
                fread(buf, spikes->wflen, sizeof(short), plxfile->fp);
                for (k = 0; k < spikes->wflen; k++) 
                    spikes->waveforms[n*spikes->wflen + k] = buf[k] / changain;
            } else {
                fseek(plxfile->fp, spikes->wflen*2 + 8, SEEK_CUR);
            }
            n++;
        }
    }
    return spikes;
}
*/

unsigned long _binary_search(FrameSet* frameset, TSTYPE ts) {
    unsigned long mid = frameset->num / 2, left = 0, right = frameset->num;
    DataFrame* frame = &(frameset->frames[mid]);

    while (right - left > 1) {
        frame = &(frameset->frames[mid]);
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