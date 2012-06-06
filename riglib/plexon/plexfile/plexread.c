#include "plexread.h"

long get_header(FILE* fp, PL_FileHeader* header, PL_ChanHeader* spikes,
    PL_EventHeader* events, PL_SlowChannelHeader* slowchans) {
    size_t readsize, data_start;

    if (fseek(fp, 0, SEEK_SET) != 0)
        return -1;

    // Read the file header
    if (fread(header, sizeof(PL_FileHeader), 1, fp) != 1)
        return -1;

    // Read the spike channel headers
    if(header->NumDSPChannels > 0) {
        readsize = fread(spikes, sizeof(PL_ChanHeader), header->NumDSPChannels, fp);
        printf("Read %lu spike channels\n", readsize);
    }

    // Read the event channel headers
    if(header->NumEventChannels > 0) {
        readsize = fread(events, sizeof(PL_EventHeader), header->NumEventChannels, fp);
        printf("Read %lu event channels\n", readsize);
    }

    // Read the slow A/D channel headers
    if(header->NumSlowChannels) {
        readsize = fread(slowchans, sizeof(PL_SlowChannelHeader), header->NumSlowChannels, fp);
        printf("Read %lu slow channels\n", readsize);
    }

    // save the position in the PLX file where data block begin
    data_start = sizeof(PL_FileHeader) + header->NumDSPChannels*sizeof(PL_ChanHeader)
                        + header->NumEventChannels*sizeof(PL_EventHeader)
                        + header->NumSlowChannels*sizeof(PL_SlowChannelHeader);

    return data_start;
}

int get_fidx(FILE* fp, long long *times, int tslen, short *types, int tylen, size_t *filepos, int flen) {
    int block = 0;
    long int offset = 0;
    long long ts;
    PL_DataBlockHeader header;

    while (fread(&header, sizeof(header), 1, fp) == 1) {
        if (block > tslen || block > tylen || block > flen)
            return -1;

        ts = (((long long) header.UpperByteOf5ByteTimestamp) << 32) | ((long long) header.TimeStamp);
        times[block] = ts;
        types[block] = header.Type;
        filepos[block] = ftell(fp);

        offset = 0;
        if (header.NumberOfWaveforms > 0 && header.NumberOfWordsInWaveform > 0) {
            offset = header.NumberOfWaveforms * header.NumberOfWordsInWaveform * 2;
            fseek(fp, offset, SEEK_CUR);
        }
        if (header.Type == 4)
            printf("Block %d, Chan %d / %d, Type %d. Skipped %d\n", block, header.Channel, header.Unit, header.Type, offset);
        block++;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc <= 1) {
        printf("Please supply a filename!\n");
        exit(1);
    }
    FILE* fp = fopen(argv[1], "rb");

    PL_FileHeader header;
    PL_ChanHeader spikes[MAX_SPIKE_CHANNELS];
    PL_EventHeader events[MAX_SPIKE_CHANNELS];
    PL_SlowChannelHeader slowchans[MAX_SPIKE_CHANNELS];

    long readsize;

    long long times[8192<<5];
    short types[8192<<5];
    size_t filepos[8192<<5];
    readsize = get_header(fp, &header, spikes, events, slowchans);
    readsize = get_fidx(fp, times, 8192<<5, types, 8192<<5, filepos, 8192<<5);
    if (readsize > 0) {
        printf("Successfully read file.\n");
        return 0;
    } else {
        return (int) readsize;
    }
    return 0;
}

