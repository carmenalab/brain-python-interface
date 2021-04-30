#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <comedilib.h>

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

comedi_t* ni;
int subdev = 0;		/* change this to your input subdevice */

extern uchar comedi_init(char* dev) {
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

extern int set_bits_in_nidaq_using_mask_and_data(int mask, int data, int base_channel) {

    int retval = comedi_dio_bitfield2(ni, subdev, mask, &data, base_channel);
    return retval;
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