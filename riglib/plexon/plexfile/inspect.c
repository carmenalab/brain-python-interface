#include "plexfile.h"
#include "plexread.h"

const char names[][128] = {
    "spikes",
    "events",
    "wideband",
    "spkc",
    "lfp",
    "analog"
};

void plx_summary(PlexFile* plxfile) {
    int i;
    printf("Plexon file %s, %0.2f seconds\n", plxfile->filename, plxfile->length);
    for (i = 0; i < ChanType_MAX; i++) {
        printf("\t%8s: %lu / %lu\n", names[i], plxfile->data[i].num, plxfile->data[i].lim);
    }
}

void plx_print_frame(DataFrame* frame) {
    printf("%s at ts=%llu, fpos=[%lu, %lu], samples=%lu, len=%lu\n", 
            names[frame->type],
            frame->ts, frame->fpos[0], frame->fpos[1],
            frame->samples, frame->nblocks
            );
}

void plx_print_frameset(FrameSet* frameset, int num) {
    int i;
    for (i = 0; i < max(0, min(num, frameset->num)); i++) {
        plx_print_frame(&(frameset->frames[i]));
    }
}

int plx_check_frames(PlexFile* plxfile, ChanType type) {
    unsigned int i, invalid = 0;
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
            if ((frame->samples /(double) freq) != tsdiff) {
                printf("Found invalid frame, ts=%f, next=%f, diff=%f, samples=%lu, expect=%f\n", 
                    frame->ts / (double) adfreq, frameset->frames[i+1].ts / adfreq, tsdiff,
                    frame->samples, frame->samples / freq);
                invalid++;
            }
        }
    }
    return invalid;
}
