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

//Send a string
uchar _send(char header, char* msg) {
    uint m, i = 0, flush;

    do {
        m = header << 8 | msg[i];
        comedi_dio_bitfield2(ni, 0, writemask, &m, 0);
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

extern int buzzer_test(int on_or_off) {
    uint val = 0; //on_or_off ? 0 : 16777215;
    //uint val = on_or_off ? 0 : 16777215;
    uint subdevice = 0;
    uint write_mask = 16777215; //pow(2, 24)-1;
    uint base_channel = 0;
    return comedi_dio_bitfield2(ni, subdevice, write_mask, &val, base_channel);
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

    // Write a '1' on every channel
    uint val = 0x100000; //0x900000; //16777215;
    uint subdevice = 0;
    uint write_mask = 16777215;
    uint base_channel = 0;
    comedi_dio_bitfield2(ni, subdevice, write_mask, &val, base_channel);

    usleep(5e5);
    //sleep(1.0);

    printf("Stopping the buzzing\n");
    uint val2 = 0;
    comedi_dio_bitfield2(ni, subdevice, write_mask, &val2, base_channel);


    closeall();
}
