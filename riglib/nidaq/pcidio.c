#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <comedilib.h>

#define SEND_DATA 0
#define SEND_MESSAGE 1
#define SEND_REGISTER 2
#define SEND_SHAPE 3

#define writemask (2 << 16 | 127<<8 | 255)

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

comedi_t* ni;
uint nsys = 0;
uint systems[32];

//Send a string
uchar _send(char header, char* msg) {
    uint m, i = 0, flush;

    do {
        m = header << 8 | msg[i];
        comedi_dio_bitfield2(ni, 0, writemask, &m, 0);
        flush = 2;
        comedi_dio_bitfield2(ni, 0, 2, &flush, 16);
        i++;
    } while (msg[i-1] != '\0');
    return 0;
}

extern uchar init(char* dev) {
    uint i, nchan;
    ni = comedi_open(dev);
    nchan = comedi_get_n_channels(ni, 0);
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
    systems[nsys] = nsys;
    _send(nsys << 3 | SEND_REGISTER, name);
    _send(nsys << 3 | SEND_SHAPE, dtype);

    return nsys++;
}

extern uchar sendData(uchar idx, char* data) {
    uint sys = systems[idx];
    return _send(sys << 3 | SEND_DATA, data);
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
    init("/dev/comedi0");
    sendMsg("This is a test!");
    //test_send();
    printf("I sent all the messages...\n");
    return 0;
}
