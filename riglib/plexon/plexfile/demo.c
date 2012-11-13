#include "plexfile.h"
#include "plexread.h"

extern char** names;

int main(int argc, char** argv) {
    if (argc <= 1) {
        printf("Please supply a filename!\n");
        exit(1);
    }
    int i, bad;
    PlexFile* plxfile = plx_open(argv[1]);
    plx_load(plxfile, false);
    plx_summary(plxfile);
    plx_print_frameset(&(plxfile->data[analog]), 100);
    plx_print_frameset(&(plxfile->data[lfp]), 10);
    for (i = 0; i < ChanType_MAX; i++) {
        printf("Checking %s...\n", names[i]);
        fflush(stdout);
        bad = plx_check_frames(plxfile, i);
        printf("Found %d bad frames!\n", bad);
    }

    plx_check_frames(plxfile, lfp);

    FILE* fp = fopen(argv[2], "wb");
    
    int chans[5] = {0, 145, 146, 147, 161};
    ContInfo* info = plx_get_continuous(plxfile, analog, atof(argv[3]), atof(argv[4]), chans, 5);
    printf("Writing all analog data, shape (%lu, %lu), t_start=%f\n", info->len, info->nchans, info->t_start);
    double data[info->len*info->nchans];
    plx_read_continuous(info, data);
    fwrite(data, sizeof(double), info->nchans*info->len, fp);

    /*
    SpikeInfo* info = plx_get_discrete(plxfile, spike, atof(argv[3]), atof(argv[4]));
    Spike spikes[info->num];
    plx_read_discrete(info, spikes);
    
    fwrite(spikes, sizeof(Spike), info->num, fp);
    printf("Wrote out %d rows of spike data\n", info->num);
    */
    plx_close(plxfile);

    return 0;
}
