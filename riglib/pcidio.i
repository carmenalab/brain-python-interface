%module pcidio
%{
extern init(char* dev);
extern sendMsg(char* msg);
extern register_sys(char* name, unsigned char ndim, unsigned short* dims);
extern sendData(unsigned char idx, double* data);
%}
