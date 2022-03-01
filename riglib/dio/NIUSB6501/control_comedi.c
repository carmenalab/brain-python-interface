#include <stdio.h>	/* for printf() */
#include <comedilib.h>
#include <unistd.h>


/*
To-do
1. error checking. 
*/

int subdev = 0;		/* change this to your input subdevice */
int range = 0;		/* more on this later */
int aref = AREF_GROUND;	/* more on this later */

int main(int argc,char *argv[])
{

	//then deal with the comedi stuff
	comedi_t *it;
	int retval;
	it = comedi_open("/dev/comedi0");
	if(it == NULL) {
		comedi_perror("comedi_open");
		return 1;
	}

	//let's get the mode
	int mode;
	sscanf(argv[1], "%d",&mode);

	int set_bit;
	int chan;
	int mask;
	int data;
	int base_channel = 0;
	int N_CHANNEL = 24;

	for(int i = 0;i<N_CHANNEL;i++){
					
			retval = comedi_dio_config(it, subdev, i, COMEDI_OUTPUT);
			//printf("channel set:%d, retbit %d \n", i, retval);

	}

	switch(mode){
		case 0: //write to channel
		    //process the arguments, eh. 
			//argv[0] is the program name, not useful


			sscanf(argv[2], "%d", &chan);
			sscanf(argv[3],"%d", &set_bit);
			printf("channel set to %d, bit set to %d \n", chan,set_bit);
			retval = comedi_dio_write(it, subdev, chan, set_bit);
			break;
			
		case 1: //write to data mask
			//


			sscanf(argv[2], "%x", &mask);
			sscanf(argv[3],"%x", &data);
			retval = comedi_dio_bitfield2(it, subdev, mask, &data, base_channel);
			printf("channel set to %x, data set to %x \n", mask,  data);

			break;


	}



		
	if(retval < 0) {
		comedi_perror("comedi_dio_write");
		return 1;
	}
	


	return 0;
}
