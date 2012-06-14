#include "plexfile.h"

PlexFile* open_plex(char* filename) {
    long readsize;
    FILE* fp;

    PlexFile* plxfile = (PlexFile*) calloc(1, sizeof(PlexFile));
    plxfile->fp = fopen(filename, "rb");
    plxfile->filename = filename;
    readsize = get_header(plxfile);
    assert(readsize > 0);

    char* cachename = _plx_cache_name(filename);
    if ((fp = fopen(cachename, "rb"))) {
        //Cache was found, let's read it
        fread(&(plxfile->spikes.num), sizeof(plxfile->spikes.num), 1, fp);
        plxfile->spikes.frames = malloc(sizeof(DataFrame)*plxfile->spikes.num);
        fread(plxfile->spikes.frames, sizeof(DataFrame), plxfile->spikes.num, fp);
        fread(&(plxfile->wideband.num), sizeof(plxfile->wideband.num), 1, fp);
        plxfile->wideband.frames = malloc(sizeof(DataFrame)*plxfile->wideband.num);
        fread(plxfile->wideband.frames, sizeof(DataFrame), plxfile->wideband.num, fp);
        fread(&(plxfile->spkc.num), sizeof(plxfile->spkc.num), 1, fp);
        plxfile->spkc.frames = malloc(sizeof(DataFrame)*plxfile->spkc.num);
        fread(plxfile->spkc.frames, sizeof(DataFrame), plxfile->spkc.num, fp);
        fread(&(plxfile->lfp.num), sizeof(plxfile->lfp.num), 1, fp);
        plxfile->lfp.frames = malloc(sizeof(DataFrame)*plxfile->lfp.num);
        fread(plxfile->lfp.frames, sizeof(DataFrame), plxfile->lfp.num, fp);
        fread(&(plxfile->analog.num), sizeof(plxfile->analog.num), 1, fp);
        plxfile->analog.frames = malloc(sizeof(DataFrame)*plxfile->analog.num);
        fread(plxfile->analog.frames, sizeof(DataFrame), plxfile->analog.num, fp);
        fread(&(plxfile->event.num), sizeof(plxfile->event.num), 1, fp);
        plxfile->event.frames = malloc(sizeof(DataFrame)*plxfile->event.num);
        fread(plxfile->event.frames, sizeof(DataFrame), plxfile->event.num, fp);
        fclose(fp);
    } else {
        plxfile->spikes.lim = 1;
        plxfile->wideband.lim = 1;
        plxfile->spkc.lim = 1;
        plxfile->lfp.lim = 1;
        plxfile->analog.lim = 1;
        plxfile->event.lim = 1;

        read_frames(plxfile);
        printf("Successfully read %lu frames\n", plxfile->nframes);
        save_cache(plxfile);
    }
    return plxfile;
}

void close_plex(PlexFile* plxfile) {
    free(plxfile->spikes.frames);
    free(plxfile->wideband.frames);
    free(plxfile->spkc.frames);
    free(plxfile->lfp.frames);
    free(plxfile->analog.frames);
    free(plxfile->event.frames);
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

void save_cache(PlexFile* plxfile) {
    char* filename = _plx_cache_name(plxfile->filename);
    printf("Saving cache to %s, %s\n", filename, plxfile->filename);
    FILE* fp = fopen(filename, "wb");
    if (fp) {
        fwrite(&(plxfile->spikes.num), sizeof(plxfile->spikes.num), 1, fp);
        fwrite(plxfile->spikes.frames, sizeof(DataFrame), plxfile->spikes.num, fp);
    fwrite(&(plxfile->wideband.num), sizeof(plxfile->wideband.num), 1, fp);
    fwrite(plxfile->wideband.frames, sizeof(DataFrame), plxfile->wideband.num, fp);
    fwrite(&(plxfile->spkc.num), sizeof(plxfile->spkc.num), 1, fp);
    fwrite(plxfile->spkc.frames, sizeof(DataFrame), plxfile->spkc.num, fp);
    fwrite(&(plxfile->lfp.num), sizeof(plxfile->lfp.num), 1, fp);
    fwrite(plxfile->lfp.frames, sizeof(DataFrame), plxfile->lfp.num, fp);
    fwrite(&(plxfile->analog.num), sizeof(plxfile->analog.num), 1, fp);
    fwrite(plxfile->analog.frames, sizeof(DataFrame), plxfile->analog.num, fp);
    fwrite(&(plxfile->event.num), sizeof(plxfile->event.num), 1, fp);
    fwrite(plxfile->event.frames, sizeof(DataFrame), plxfile->event.num, fp);
    fclose(fp);
    } else {
         printf("Unable to save cache!\n");
    }
    free(filename);
}

long get_header(PlexFile* plxfile) {
    size_t readsize, data_start;

    if (fseek(plxfile->fp, 0, SEEK_SET) != 0)
        return -1;

    // Read the file header
    if (fread(&(plxfile->header), sizeof(PL_FileHeader), 1, plxfile->fp) != 1)
        return -1;

    // Read the spike channel headers
    if(plxfile->header.NumDSPChannels > 0) {
        readsize = fread(&(plxfile->chan_info), sizeof(PL_ChanHeader), plxfile->header.NumDSPChannels, plxfile->fp);
        printf("Read %lu spike channels\n", readsize);
    }

    // Read the event channel headers
    if(plxfile->header.NumEventChannels > 0) {
        readsize = fread(&(plxfile->event_info), sizeof(PL_EventHeader), plxfile->header.NumEventChannels, plxfile->fp);
        printf("Read %lu event channels\n", readsize);
    }

    // Read the slow A/D channel headers
    if(plxfile->header.NumSlowChannels) {
        readsize = fread(&(plxfile->cont_info), sizeof(PL_SlowChannelHeader), plxfile->header.NumSlowChannels, plxfile->fp);
        printf("Read %lu slow channels\n", readsize);
    }

    // save the position in the PLX file where data block begin
    data_start = sizeof(PL_FileHeader) + plxfile->header.NumDSPChannels*sizeof(PL_ChanHeader)
                        + plxfile->header.NumEventChannels*sizeof(PL_EventHeader)
                        + plxfile->header.NumSlowChannels*sizeof(PL_SlowChannelHeader);

    return data_start;
}
