%module control_comedi_swig
%{
extern unsigned char comedi_init(char* dev);
extern int set_bits_in_nidaq_using_mask_and_data(int mask, int data);
%}

extern unsigned char comedi_init(char* dev);
extern int set_bits_in_nidaq_using_mask_and_data(int mask, int data);