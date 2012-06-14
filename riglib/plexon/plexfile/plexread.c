#include "plexread.h"

unsigned long int _binary_search(FrameSet* frameset, TSTYPE ts) {
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

void plx_read_continuous(FILE* fp, FrameSet* frameset, int tsfreq, int chanfreq, int gain,
    TSTYPE start, TSTYPE stop, int* chans, int nchans, 
    double* data) {

    int i, f, c, t, _lchan, _chan, stride;
    DataFrame* frame = &(frameset->frames[0]);
    double t_off = frame->ts / (double) tsfreq;
    short buf[MAX_SAMPLES_PER_WAVEFORM];
    long t_start;

    start = start == 0 ? start : _binary_search(frameset, start);
    stop = stop == 0 ? frameset->num : _binary_search(frameset, stop);

    if (chans == NULL) {
        chans = malloc(sizeof(int) * frame->nblocks);
        for (i = 0; i < frame->nblocks; i++)
            chans[i] = 1;
        nchans = frame->nblocks;
    } else {
        //Compute the channel differential
        _lchan = chans[0];
        for (i = 1; i < nchans; i++) {
            _chan = chans[i];
            chans[i] = chans[i] - _lchan;
            _lchan = _chan;
        }
    }

    for (f = start; f < stop; f++) {
        frame = &(frameset->frames[f]);
        t_start = (frame->ts / (double) tsfreq - t_off) * chanfreq;
        stride = frame->samples * sizeof(short) + sizeof(PL_DataBlockHeader);
        fseek(fp, frame->fpos[0]+sizeof(PL_DataBlockHeader), SEEK_SET);

        for (c = 0; c < nchans; c++) {
            fread(buf, sizeof(short), frame->samples, fp);
            fseek(fp, stride*chans[c], SEEK_CUR);
            for (t = 0; t < frame->samples; t++)
                data[(t+t_start)*nchans + c] = buf[t] / (double) gain;
        }
    }

    if (nchans == frame->nblocks)
        free(chans);
}

ContData* plx_readall_analog(PlexFile* plxfile) {
    PL_SlowChannelHeader chaninfo = plxfile->cont_info[plxfile->header.NumDSPChannels * 3];
    ContData* data = malloc(sizeof(ContData));
    int tsfreq = plxfile->header.ADFrequency;

    data->freq = chaninfo.ADFreq;
    data->t_start = plxfile->analog.frames[0].ts;
    data->nchans = plxfile->analog.frames[0].nblocks;
    data->len = (plxfile->analog.frames[plxfile->analog.num-1].ts - data->t_start) / (double) tsfreq * data->freq;
    data->len += plxfile->analog.frames[plxfile->analog.num-1].samples;

    data->data = calloc(data->len * data->nchans, sizeof(double));

    plx_read_continuous(plxfile->fp, &(plxfile->analog), tsfreq, chaninfo.ADFreq, chaninfo.Gain,
        0, 0, NULL, 0, data->data);

    return data;
}

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