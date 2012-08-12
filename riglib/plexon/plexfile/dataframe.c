#include "plexfile.h"
#include "dataframe.h"

int lastchan = 0;

void plx_get_frames(PlexFile* plxfile) {
    long int laststart;
    int nchans = plxfile->header.NumDSPChannels;
    unsigned long int start_pos = ftell(plxfile->fp);
    double end_pos;
    SimpleDatablock header;
    DataFrame* frame;
    plxfile->nframes = 0;

    fseek(plxfile->fp, 0L, SEEK_END);
    end_pos = (double) ftell(plxfile->fp);
    fseek(plxfile->fp, start_pos, SEEK_SET);

    printf("Reading frames...\n");
    laststart = _plx_read_datablock(plxfile->fp, nchans, &header);
    _plx_new_frame(&header, laststart, plxfile, &frame);
    while ((laststart = _plx_read_datablock(plxfile->fp, nchans, &header)) > 0) {
        if (header.ts != frame->ts || header.type != frame->type || 
            (unsigned long) header.samples != frame->samples) {

            frame->fpos[1] = laststart;
            _plx_new_frame(&header, laststart, plxfile, &frame);
            (plxfile->nframes)++;

            if ((plxfile->nframes % 100) == 0) {
                printf("%0.2f%%...      \r", ((double)ftell(plxfile->fp) / end_pos) * 100);
                fflush(stdout);
            }
        }
        (frame->nblocks)++;
        if (( frame->type == wideband || frame->type == spkc || 
              frame->type == lfp || frame->type == analog) && 
            lastchan+1 != header.chan) {
            printf("Error, channels not in order: %d -- ts=%llu, type=%d, chan=%d\n", lastchan, header.ts, header.type, header.chan);
            exit(1);
        }
        lastchan = header.chan;
    }
    frame->fpos[1] = ftell(plxfile->fp);
}

void _plx_new_frame(SimpleDatablock* header, unsigned long start, PlexFile* plxfile, DataFrame** frame) {
    FrameSet* frameset = &(plxfile->data[header->type]);
    if (frameset->lim <= ((frameset->num)+1)) {
        frameset->lim *= 2;
#ifdef DEBUG
        printf("Allocating %lu bytes for frameset...\n", sizeof(DataFrame)*frameset->lim);
#endif
        frameset->frames = realloc(frameset->frames, sizeof(DataFrame)*frameset->lim);
        if (frameset->frames == NULL) {
            printf("Unable to allocate memory...\n");
            exit(1);
        }
    }
    *frame = &(frameset->frames[frameset->num++]);
    (*frame)->ts = header->ts;
    (*frame)->type = header->type;
    (*frame)->samples = header->samples;
    (*frame)->fpos[0] = start;
    (*frame)->nblocks = 0;
    lastchan = 0;
    //printf("New frame: ts=%llu, type=%d, samples=%d\n", header->ts, header->type, header->samples);
}

long int _plx_read_datablock(FILE* fp, int nchannels, SimpleDatablock* block) {
    size_t readsize;
    PL_DataBlockHeader header;
    long int start = (long int) ftell(fp);

    readsize = fread(&header, sizeof(PL_DataBlockHeader), 1, fp);
    if (readsize != 1)
        return -1;

    block->ts = (((TSTYPE) header.UpperByteOf5ByteTimestamp) << 32) | ((TSTYPE) header.TimeStamp);
    block->samples = header.NumberOfWaveforms * header.NumberOfWordsInWaveform;
    block->chan = header.Channel;
    block->unit = header.Unit;
    if (header.Type == PL_SingleWFType)
        block->type = spike;
    else if (header.Type == PL_ExtEventType)
        block->type = event;
    else if (header.Type == PL_ADDataType) {
        block->chan = header.Channel % nchannels + 1;
        switch (header.Channel / nchannels) {
            case 0:
                block->type = wideband;
                break;
            case 1:
                block->type = spkc;
                break;
            case 2:
                block->type = lfp;
                break;
            case 3:
                block->type = analog;
                break;
        }
    }
    if (block->samples > 0)
        fseek(fp, block->samples*sizeof(short), SEEK_CUR);

    //printf("Block read: ts=%llu, chan=%d, hchan=%d, samples=%d, type=%d,%d\n", block->ts, block->chan, header.Channel, block->samples, header.Type, block->type);

    return start;
}