#include "plexfile.h"

extern PlexFile* plx_open(char* filename) {
    long readsize;

    PlexFile* plxfile = (PlexFile*) calloc(1, sizeof(PlexFile));
    plxfile->fp = fopen(filename, "rb");
    if (!plxfile->fp)
        return NULL;

    plxfile->filename = calloc(strlen(filename), sizeof(char));
    strcpy(plxfile->filename, filename);
    readsize = plx_get_header(plxfile);
    assert(readsize > 0);

    return plxfile;
}

extern void plx_load(PlexFile* plxfile, bool recache) {
    int i;
    FILE* fp;
    size_t readsize;

    char* cachename = _plx_cache_name(plxfile->filename);
    if ((fp = fopen(cachename, "rb")) && !recache) {
        printf("Found cache, opening...\n");
        for (i = 0; i < ChanType_MAX; i++) {
            readsize = fread(&(plxfile->data[i].num), sizeof(plxfile->data[i].num), 1, fp);
            assert(readsize == 1);
            plxfile->data[i].frames = malloc(sizeof(DataFrame)*plxfile->data[i].num);
            readsize = fread(plxfile->data[i].frames, sizeof(DataFrame), plxfile->data[i].num, fp);
            assert(readsize == plxfile->data[i].num);
        }
        fclose(fp);
    } else {
        for (i = 0; i < ChanType_MAX; i++) 
            plxfile->data[i].lim = 1;
        plx_get_frames(plxfile);
        printf("Successfully read %lu frames\n", plxfile->nframes);
        plx_save_cache(plxfile);
    }

    plxfile->has_cache = true;
}

extern void plx_close(PlexFile* plxfile) {
    int i;
    if (plxfile->has_cache) {
        for (i = 0; i < ChanType_MAX; i++)
            free(plxfile->data[i].frames);
    }

    free(plxfile->filename);
    fclose(plxfile->fp);
    free(plxfile);
}

char* _plx_cache_name(char* filename) {
    char* fullfile = realpath(filename, NULL);
    char* cachename = calloc(strlen(fullfile)+3, sizeof(char));
    char* filepart = strrchr(fullfile, '/');
    assert(filepart != NULL);
    strncpy(cachename, fullfile, filepart-fullfile+1);
    filepart[strlen(filepart)-4] = '\0';
    sprintf(cachename, "%s.%s.cache", cachename, filepart+1);
    free(fullfile);
    return cachename;
}

void plx_save_cache(PlexFile* plxfile) {
    int i;
    char* filename = _plx_cache_name(plxfile->filename);
    printf("Saving cache to %s, %s\n", filename, plxfile->filename);
    FILE* fp = fopen(filename, "wb");
    if (fp) {
        for (i = 0; i < ChanType_MAX; i++) {
            fwrite(&(plxfile->data[i].num), sizeof(plxfile->data[i].num), 1, fp);
            fwrite(plxfile->data[i].frames, sizeof(DataFrame), plxfile->data[i].num, fp);
        }
        fclose(fp);
    } else {
        printf("Unable to save cache!\n");
    }
    free(filename);
}

long plx_get_header(PlexFile* plxfile) {
    int i;
    size_t readsize, data_start;

    if (fseek(plxfile->fp, 0, SEEK_SET) != 0)
        return -1;

    // Read the file header
    if (fread(&(plxfile->header), sizeof(PL_FileHeader), 1, plxfile->fp) != 1)
        return -1;

    // Read the spike channel headers
    if(plxfile->header.NumDSPChannels > 0) {
        readsize = fread(&(plxfile->chan_info), sizeof(PL_ChanHeader), plxfile->header.NumDSPChannels, plxfile->fp);
        printf("Found %lu spike channels, ", readsize);
    }

    // Read the event channel headers
    if(plxfile->header.NumEventChannels > 0) {
        readsize = fread(&(plxfile->event_info), sizeof(PL_EventHeader), plxfile->header.NumEventChannels, plxfile->fp);
        printf("%lu event channels, ", readsize);
    }

    // Read the slow A/D channel headers
    if(plxfile->header.NumSlowChannels) {
        //assert(plxfile->header.NumSlowChannels >= 3*plxfile->header.NumDSPChannels);
        readsize = fread(&(plxfile->cont_head), sizeof(PL_SlowChannelHeader), plxfile->header.NumSlowChannels, plxfile->fp);
        printf("and %lu slow channels\n", readsize);
        for (i = 0; i < 4; i++)
            plxfile->cont_info[i] = &(plxfile->cont_head[i*plxfile->header.NumDSPChannels]);
    }

    plxfile->length = plxfile->header.LastTimestamp / plxfile->header.ADFrequency;

    // save the position in the PLX file where data block begin
    data_start = sizeof(PL_FileHeader) + plxfile->header.NumDSPChannels*sizeof(PL_ChanHeader)
                        + plxfile->header.NumEventChannels*sizeof(PL_EventHeader)
                        + plxfile->header.NumSlowChannels*sizeof(PL_SlowChannelHeader);

    return data_start;
}
