typedef struct {
    unsigned long long ts;
    unsigned short chan;
    unsigned short unit;
} __attribute__((packed)) Spike;

typedef struct {
    int chan;
    int unit;
} Channel;

typedef unsigned int uint;

unsigned int _hash_chan(unsigned short chan, unsigned short unit);
extern void set_channels(char* bufchan, size_t clen);
extern void binspikes(float length, char* bufspikes, size_t slen, unsigned int* counts, int countlen);