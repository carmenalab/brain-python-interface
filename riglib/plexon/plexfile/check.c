#include "plexfile.h"

int check_frameset(FrameSet* frameset, int freq, int adfreq) {
    int i, invalid = 0;
    double tsdiff = 0;
    DataFrame* frame;
    if (frameset->num > 0) {
        for (i = 0; i < frameset->num-1; i++) {
            frame = &(frameset->frames[i]);
            tsdiff = (frameset->frames[i+1].ts - frame->ts) / ((double) adfreq);
            assert(tsdiff > 0);
            if (frame->samples /(double) freq != tsdiff) {
                printf("Found invalid block, ts=%f, next=%f, diff=%f, samples=%lu, expect=%f\n", 
                    frame->ts / (double) adfreq, frameset->frames[i+1].ts / (double) adfreq, tsdiff,
                    frame->samples, frame->samples / (double) freq);
                invalid++;
            }
        }
    }
    return invalid;
}
