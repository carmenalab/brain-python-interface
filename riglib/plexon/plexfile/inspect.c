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
    for (i = 0; i < ChanType_MAX; i++) {
        printf("Found %8s: %lu / %lu\n", names[i], plxfile->data[i].num, plxfile->data[i].lim);
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

    if (num < 0)
        num = frameset->num;

    for (i = 0; i < min(num, frameset->num); i++) {
        plx_print_frame(&(frameset->frames[i]));
    }
}

int main(int argc, char** argv) {
    if (argc <= 1) {
        printf("Please supply a filename!\n");
        exit(1);
    }
    int i, bad;
    PlexFile* plxfile = plx_open(argv[1]);
    plx_summary(plxfile);
    plx_print_frameset(&(plxfile->data[lfp]), 100);
    plx_print_frameset(&(plxfile->data[wideband]), 10);
    for (i = 0; i < ChanType_MAX; i++) {
        printf("Checking %s...\n", names[i]);
        fflush(stdout);
        bad = plx_check_frames(plxfile, i);
        printf("Found %d bad frames!\n", bad);
    }
    unsigned long idx = _binary_search(&(plxfile->data[lfp]), 0);
    printf("Found at %lu, ", idx);
    plx_print_frame(&(plxfile->data[lfp].frames[idx]));
        
/*
    FILE* fp = fopen(argv[2], "wb");
    //ContData* data = plx_readall_analog(plxfile);
    //printf("Writing all analog data, shape (%lu, %lu)\n", data->len, data->nchans);
    //fwrite(&(data->data[i*data->nchans]), sizeof(double), data->nchans*data->len, fp);
    //free(data->data);
    //free(data);

    SpikeData* data = plx_readall_spikes(plxfile, false);
    fwrite(data->spike, sizeof(Spike), data->num, fp);
    printf("Wrote out %d rows of spike data\n", data->num);
*/
    plx_close(plxfile);

    return 0;
}
