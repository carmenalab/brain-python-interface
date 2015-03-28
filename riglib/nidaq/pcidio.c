#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <comedilib.h>

#define SEND_DATA 0
#define SEND_MESSAGE 1
#define SEND_REGISTER 2
#define SEND_SHAPE 3
#define SEND_ROW 4
#define SEND_ROWBYTE 5

#define writemask (2 << 16 | 127<<8 | 255)

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

comedi_t* ni;
uint nsys = 0;
uint rowcount[32];

// Call signature for comedi_dio_bitfield2 (in C)
// int comedi_dio_bitfield2(   comedi_t * device,
//     unsigned int subdevice,
//     unsigned int write_mask,
//     unsigned int * bits,
//     unsigned int base_channel);

//Send a string
uchar _send(char header, char* msg) {
    uint m, i = 0, flush;

    do {
        // "Load" the data message; implicity sets the omniplex Strobe pin to 0
        m = header << 8 | msg[i];
        comedi_dio_bitfield2(ni, 0, writemask, &m, 0);

        // Turn on the Strobe pin
        flush = 2;
        comedi_dio_bitfield2(ni, 0, 2, &flush, 16);
    } while (msg[i++] != '\0');
    return 0;
}

extern uchar init(char* dev) {
    uint i, nchan;
    ni = comedi_open(dev);
    if (ni == NULL) {
        printf("Error opening device\n");
        return -1;
    }
    nchan = comedi_get_n_channels(ni, 0);
    printf("found comedi system with %d channels\n", nchan);
    for (i=0; i < nchan; i++) {
        if (comedi_dio_config(ni, 0, i, COMEDI_OUTPUT) < 0)
            return -1;
    }
    return 0;
}

extern uchar sendMsg(char* msg) {
    return _send(SEND_MESSAGE, msg);
}

extern uint register_sys(char* name, char* dtype) {
    _send(nsys << 3 | SEND_REGISTER, name);
    _send(nsys << 3 | SEND_SHAPE, dtype);
    rowcount[nsys] = 0;
    
    return nsys++;
}

extern uchar sendData(uchar idx, char* data) {
    return _send(idx << 3 | SEND_DATA, data);
}

extern uchar sendRow(uchar idx, uint row) {
    char* msg = (char*) &row;
    uint i, m, flush = 2;

    for (i = 0; i < sizeof(uint); i++) {
        m = (idx << 3 | SEND_ROW) << 8 | msg[i];
        comedi_dio_bitfield2(ni, 0, writemask, &m, 0);
        comedi_dio_bitfield2(ni, 0, 2, &flush, 16);
    }
    return 0;
}
extern uchar sendRowByte(uchar idx) {
    uint flush = 2, msg = (idx << 3 | SEND_ROWBYTE) << 8 | (255 & rowcount[idx]);
    comedi_dio_bitfield2(ni, 0, writemask, &msg, 0);
    comedi_dio_bitfield2(ni, 0, 2, &flush, 16);
    rowcount[idx]++;
    return 0;
}
extern uchar sendRowCount(uchar idx) {
    return sendRow(idx, rowcount[idx]++);
}

extern int rstart(uint start) {
    uint val = start ? 0 : 16;
    return comedi_dio_bitfield2(ni, 0, 16, &val, 16);
}

extern uchar closeall(void) {
    return comedi_close(ni);
}

void test_bits() {
    uint i, m, flush;
    for (i = 0; i < 15; i++) {
        m = i;
        comedi_dio_bitfield2(ni, 0, writemask, &m, 0);
        flush = 2;
        comedi_dio_bitfield2(ni, 0, 2, &flush, 16);
        usleep(100000);
    }
}

int main(int argc, char** argv) {
    //init("/dev/comedi0");
    //sendMsg("This is a test!");
    uint t = 26729;
    char* m = (char*) &t;
    printf("%c, %c, %c, %c\n", m[0], m[1], m[2], m[3]);

    printf("I sent all the messages...\n");
    return 0;
}
