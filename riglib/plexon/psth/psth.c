#include <stdio.h>
#include <string.h>
#include "psth.h"

uint chan_set = 0;
uint chanmap[8192];

uint _hash_chan(unsigned short chan, unsigned short unit) {
    uint ichan = (uint) chan, iunit = (uint) unit;
    return (((ichan & 1023) << 3) | (iunit & 7));
}

extern void set_channels(char* bufchan, size_t clen) {
    int i, idx, num = clen / sizeof(Channel);
    Channel* channels = (Channel*) bufchan;
    for (i = 0; i < num; i++) {
        idx = _hash_chan((unsigned short) channels[i].chan, (unsigned short) channels[i].unit);
        chanmap[idx] = i+1;
        //printf("Setting hash(%d, %d) = %d to %d\n", channels[i].chan, channels[i].unit, idx, i+1);
    }
    chan_set = i;
}

extern void binspikes(double length, char* bufspikes, size_t slen, unsigned int* counts, int countlen) {
    uint i, idx, num = slen / sizeof(Spike);
    Spike* spikes = (Spike*) bufspikes;
    double curtime = spikes[num-1].ts, tdiff;

    //Since spike times are only vaguely ordered, find the largest timestamp
    for (i = num-1; i > num-100 && i > 0; i--) {
        if (spikes[i].ts > curtime)
            curtime = spikes[i].ts;
    }

    //Clear the buffer
    for (i = 0; i < countlen; i++) {
        counts[i] = 0;
    }

    //Start at beginning, counting the spikes
    for (i = num-1; i > 0; i--) {
        tdiff = curtime - spikes[i].ts;
        if (tdiff > 1.9*length) {
            //Only exit out when DOUBLE the length of time has been counted (to ensure catching everything)
            //printf("tdiff = %f, curtime=%f, time=%f\n", curtime-spikes[i].ts, curtime, spikes[i].ts);
            return;
        }

        idx = _hash_chan(spikes[i].chan, spikes[i].unit);
        //printf("%d(%d, %d): %d \n", idx, spikes[i].chan, spikes[i].unit, chanmap[idx]);
        if (chanmap[idx] > 0 && chanmap[idx]-1 < countlen && tdiff <= length) {
            counts[chanmap[idx]-1]++;
            //printf("found (%d,%d), incrementing %d\n", spikes[i].chan, spikes[i].unit, chanmap[idx]-1);
        }
    }
    printf("Buffer underrun\n", curtime - spikes[0].ts, length);
}
