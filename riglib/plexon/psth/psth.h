typedef struct {
    double ts;
    int chan;
    int unit;
} Spike;

typedef struct {
    int chan;
    int unit;
} Channel;

typedef unsigned int uint;

unsigned int _hash_chan(unsigned short chan, unsigned short unit);
extern void set_channels(char* bufchan, size_t clen);
extern void binspikes(double length, char* bufspikes, size_t slen, unsigned int* counts, int countlen);