#include "plexread.h"

unsigned long int _binary_search(FrameSet* frameset, TSTYPE ts) {
    return 0;
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
        for (c = 0; c < frame->nblocks; c++) {
            fread(buf, sizeof(short), frame->samples, fp);
            fseek(fp, stride, SEEK_CUR);
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

    data->data = malloc(sizeof(double) * data->len * data->nchans);

    plx_read_continuous(plxfile->fp, &(plxfile->analog), tsfreq, chaninfo.ADFreq, chaninfo.Gain,
        0, 0, NULL, 0, data->data);

    return data;
}