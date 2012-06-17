%module pcidio
%{
extern unsigned char init(char* dev);
extern unsigned char closeall();
extern unsigned char sendMsg(char* msg);
extern unsigned int register_sys(char* name, char* dtype);
extern unsigned char sendData(unsigned char idx, char* data);
extern unsigned char sendRowCount(unsigned char idx);
extern unsigned char sendRowByte(unsigned char idx);
extern unsigned char rstart(unsigned int start);
%}
extern unsigned char init(char* dev);
extern unsigned char closeall();
extern unsigned char sendMsg(char* msg);
extern unsigned int register_sys(char* name, char* dtype);
extern unsigned char sendData(unsigned char idx, char* data);
extern unsigned char sendRowCount(unsigned char idx);
extern unsigned char sendRowByte(unsigned char idx);
extern unsigned char rstart(unsigned int start);