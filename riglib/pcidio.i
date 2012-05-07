%module pcidio
%{
extern unsigned char init(char* dev);
extern unsigned char sendMsg(char* msg);
extern unsigned int register_sys(char* name, unsigned char ndim, unsigned short* dims);
extern unsigned char sendData(unsigned char idx, double* data);
%}
extern unsigned char init(char* dev);
extern unsigned char sendMsg(char* msg);
extern unsigned int register_sys(char* name, unsigned char ndim, unsigned short* dims);
extern unsigned char sendData(unsigned char idx, double* data);
