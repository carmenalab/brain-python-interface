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

void plx_print_frame(DataFrame* frame, FILE* fp) {
    SimpleDatablock data;

    printf("Frame at ts=%f, fpos=%lu, samples=%u, len=%u", 
            frame->ts / 40000., frame->fpos, frame->samples, frame->nblocks
            );

    if (fp != NULL) {
        fseek(fp, frame->fpos, SEEK_SET);
        _plx_read_datablock(fp, frame->nblocks, &data);
        printf(", unit=%d(%c)\n", data.unit & 255, data.unit & 255);
    } else {
        printf("\n");
    }
}

void plx_print_frameset(PlexFile* plxfile, ChanType type, int start, int stop, bool detail) {
    FrameSet* frameset = &(plxfile->data[type]);
    int i;
    for (i = start; i < max(0, min(stop, frameset->num)); i++) {
        if (detail) {
            plx_print_frame(&(frameset->frames[i]), plxfile->fp);
        } else {
            plx_print_frame(&(frameset->frames[i]), NULL);
        }
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
                printf("Found invalid frame, ts=%f, next=%f, diff=%f, samples=%u, expect=%f\n", 
                    frame->ts / adfreq, frameset->frames[i+1].ts / adfreq, tsdiff,
                    frame->samples, frame->samples / freq);
                invalid++;
            }
        }
    }
    return invalid;
}
