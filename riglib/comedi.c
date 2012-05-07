#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <comedilib.h>
#define MAX_DIMS 32

#define SEND_DATA 0
#define SEND_MESSAGE 1
#define SEND_REGISTER 2
#define SEND_SHAPE 3

#define writemask (255 << 16 | 127<<8 | 255)

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

comedi_t* ni;
uint nsys = 0;
uint systems[8];

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
//Send a data array
uchar _senddata(char header, uint dlen, double* data) {
    uint i, m, flush;
    uchar dstr[sizeof(double)*dlen];

    strncpy(dstr, (uchar*)data, sizeof(double)*dlen);
    for (i = 0; i < sizeof(double)*dlen; i++) {
        m = header << 8 | dstr[i];
        comedi_dio_bitfield2(ni, 0, writemask, &m, 0);
        flush = 2;
        comedi_dio_bitfield2(ni, 0, 2, &flush, 16);
    }
    return 0;
}

//Send data shape
uchar _sendshape(char header, uchar ndim, ushort* dims) {
    uint i, m, flush;
    uchar dstr[sizeof(short)*ndim];

    strncpy(dstr, (uchar*)dims, sizeof(ushort)*ndim);
    for (i = 0; i < sizeof(ushort)*ndim; i++) {
        m = header << 8 | dstr[i];
        comedi_dio_bitfield2(ni, 0, writemask, &m, 0);
        flush = 2;
        comedi_dio_bitfield2(ni, 0, 2, &flush, 16);
    }
    return 0;
}

uchar init(char* dev) {
    uint i, nchan;
    ni = comedi_open(dev);
    nchan = comedi_get_n_channels(ni, 0);
    for (i=0; i < nchan; i++) {
        if (comedi_dio_config(ni, 0, i, COMEDI_OUTPUT) < 0)
            return -1;
    }
    return 0;
}

uchar sendMsg(char* msg) {
    return _send(SEND_MESSAGE, msg);
}

uint register_sys(char* name, uchar ndim, ushort dims[MAX_DIMS]) {
    uint i;
    uint dlen = (uint) dims[0];

    for (i = 1; i < ndim; i++)
        dlen *= dims[i];
    
    systems[nsys] = dlen;
    _send(nsys << 3 | SEND_REGISTER, name);
    _sendshape(SEND_SHAPE, ndim, dims);

    return nsys++;
}

uchar sendData(uchar idx, double* data) {
    uint dlen = systems[idx];
    return _senddata(SEND_DATA, dlen, data);
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

int main(int argc, char* argv) {
    init("/dev/comedi0");
    sendMsg("This is a test!");
    sleep(1);
    ushort dims[] = {2, 3};
    double data[] = {1, 2, 3, 4, 5, 6};
    printf("Register motion system at idx %d\n", register_sys("motion", 2, dims));
    sleep(1);
    printf("Send simulated data:%d\n", sendData(0, data));

    //test_send();
    printf("I sent all the messages...\n");
}