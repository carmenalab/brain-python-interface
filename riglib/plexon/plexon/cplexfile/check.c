#include "plexfile.h"

int plx_check_frames(PlexFile* plxfile, ChanType type) {
    int i, invalid = 0;
    double tsdiff = 0;
    DataFrame* frame;
    FrameSet* frameset = &(plxfile->data[type]);
    double adfreq = (double) plxfile->header.ADFrequency;
    double freq;

    switch(type) {
        case wideband: case spkc: case lfp: case analog:
            freq = (double) plxfile->cont_info[type - wideband]->ADFreq;
            break;
        default:
            return -1;
    }

    if (frameset->num > 0) {
        for (i = 0; i < frameset->num-1; i++) {
            frame = &(frameset->frames[i]);
            tsdiff = (frameset->frames[i+1].ts - frame->ts) / adfreq;
            assert(tsdiff > 0);
            if (frame->samples /(double) freq != tsdiff) {
                printf("Found invalid frame, ts=%f, next=%f, diff=%f, samples=%lu, expect=%f\n", 
                    frame->ts / (double) adfreq, frameset->frames[i+1].ts / adfreq, tsdiff,
                    frame->samples, frame->samples / freq);
                invalid++;
            }
        }
    }
    return invalid;
}
