#include "plexfile.h"
#include "dataframe.h"

int lastchan = 0;

long int _read_datablock(FILE* fp, int nchannels, SimpleDatablock* block) {
    size_t readsize;
    PL_DataBlockHeader header;
    long int start = (long int) ftell(fp);

    readsize = fread(&header, sizeof(PL_DataBlockHeader), 1, fp);
    if (readsize != 1)
        return -1;

    block->ts = (((unsigned long long) header.UpperByteOf5ByteTimestamp) << 32) | ((unsigned long long) header.TimeStamp);
    block->samples = header.NumberOfWaveforms * header.NumberOfWordsInWaveform;
    block->chan = header.Channel;
    block->unit = header.Unit;
    if (header.Type == PL_SingleWFType)
        block->type = chantype_spike;
    else if (header.Type == PL_ExtEventType)
        block->type = chantype_event;
    else if (header.Type == PL_ADDataType) {
        block->chan = header.Channel % nchannels + 1;
        switch (header.Channel / nchannels) {
            case 0:
                block->type = chantype_wideband;
                break;
            case 1:
                block->type = chantype_spkc;
                break;
            case 2:
                block->type = chantype_lfp;
                break;
            case 3:
                block->type = chantype_analog;
                break;
        }
    }
    if (block->samples > 0)
        fseek(fp, block->samples*sizeof(short), SEEK_CUR);

    //printf("Block read: ts=%llu, chan=%d, hchan=%d, samples=%d, type=%d,%d\n", block->ts, block->chan, header.Channel, block->samples, header.Type, block->type);

    return start;
}

void _new_frame(SimpleDatablock* header, int start, PlexFile* plxfile, DataFrame** frame) {
    FrameSet* frameset;
    switch (header->type) {
        case chantype_spike:
            frameset = &(plxfile->spikes);
            break;
        case chantype_wideband:
            frameset = &(plxfile->wideband);
            break;
        case chantype_spkc:
            frameset = &(plxfile->spkc);
            break;
        case chantype_lfp:
            frameset = &(plxfile->lfp);
            break;
        case chantype_analog:
            frameset = &(plxfile->analog);
            break;
        case chantype_event:
            frameset = &(plxfile->event);
            break;
        default:
            printf("Unknown event type, how is this possible?!\n");
            exit(1);
    }
    if (frameset->lim <= ((frameset->num)+1)) {
        frameset->lim *= 2;
        //printf("Allocating %lu bytes for frameset...\n", sizeof(DataFrame)*frameset->lim);
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
}

void read_frames(PlexFile* plxfile) {
    long int laststart;
    unsigned long int start_pos = ftell(plxfile->fp);
    double end_pos;
    SimpleDatablock header;
    DataFrame* frame;
    plxfile->nframes = 0;

    fseek(plxfile->fp, 0, SEEK_END);
    end_pos = (double) ftell(plxfile->fp);
    fseek(plxfile->fp, start_pos, SEEK_SET);

    printf("Reading frames...\n");
    _new_frame(&header, _read_datablock(plxfile->fp, plxfile->header.NumDSPChannels, &header), plxfile, &frame);
    while ((laststart = _read_datablock(plxfile->fp, plxfile->header.NumDSPChannels, &header)) > 0) {
        if (header.ts != frame->ts ||
            header.type != frame->type ||
            header.samples != frame->samples) {

            frame->fpos[1] = laststart;
            _new_frame(&header, laststart, plxfile, &frame);
            (plxfile->nframes)++;
            if (((plxfile->nframes) % 100) == 0) 
                printf("%0.1f%%...      \r", ftell(plxfile->fp) / end_pos * 100);
        }
        (frame->nblocks)++;
        if (( 2<= frame->type && frame->type <= 5) && lastchan+1 != header.chan) {
            printf("Damn, channels not in order: %d -- ts=%llu, type=%d, chan=%d, \n", lastchan, header.ts, header.type, header.chan);
            exit(1);
        }
        lastchan = header.chan;
    }
    frame->fpos[1] = ftell(plxfile->fp);
}