#include "plexfile.h"
#include "plexread.h"

char names[][128] = {
    "",
    "spikes",
    "wideband",
    "spkc",
    "lfp",
    "analog",
    "events"
};

void print_plxfile(PlexFile* plxfile) {
    printf("Plexfile spikes:   %lu / %lu\n", plxfile->spikes.num,  plxfile->spikes.lim);
    printf("Plexfile wideband: %lu / %lu\n", plxfile->wideband.num,  plxfile->wideband.lim);
    printf("Plexfile spkc:     %lu / %lu\n", plxfile->spkc.num,  plxfile->spkc.lim);
    printf("Plexfile lfp:      %lu / %lu\n", plxfile->lfp.num,  plxfile->lfp.lim);
    printf("Plexfile analog:   %lu / %lu\n", plxfile->analog.num,  plxfile->analog.lim);
    printf("Plexfile events:   %lu / %lu\n", plxfile->event.num,  plxfile->event.lim);
}

void print_frameset(FrameSet* frameset, int num) {
    int i;
    DataFrame* frame;

    if (num < 0)
        num = frameset->num;

    for (i = 0; i < num; i++) {
        frame = &(frameset->frames[i]);
        printf("%s at ts=%llu, fpos=[%lu, %lu], samples=%lu, len=%lu\n", 
            names[frame->type],
            frame->ts, frame->fpos[0], frame->fpos[1],
            frame->samples, frame->nblocks
            );
    }
}

int main(int argc, char** argv) {
    if (argc <= 1) {
        printf("Please supply a filename!\n");
        exit(1);
    }
    int i;
    PlexFile* plxfile = open_plex(argv[1]);
    print_plxfile(plxfile);
    print_frameset(&(plxfile->lfp), 100);
    printf("Found %d invalid analog frames\n", check_frameset(&(plxfile->analog), 1000, plxfile->header.ADFrequency));
    printf("Found %d invalid spkc frames\n", check_frameset(&(plxfile->spkc), 40000, plxfile->header.ADFrequency));
    printf("Found %d invalid lfp frames\n", check_frameset(&(plxfile->lfp), 1000, plxfile->header.ADFrequency));

    ContData* data = plx_readall_analog(plxfile);
    FILE* fp = fopen(argv[2], "wb");
    printf("Writing all analog data, shape (%lu, %lu)\n", data->len, data->nchans);
    fwrite(&(data->data[i*data->nchans]), sizeof(double), data->nchans*data->len, fp);
    free(data->data);
    free(data);

    close_plex(plxfile);
    return 0;
}