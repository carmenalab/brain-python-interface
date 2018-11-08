import comedi
import time
com = comedi.comedi_open("/dev/comedi0")
time.sleep(0.010)
print("starting file recording")
comedi.comedi_dio_bitfield2(com,0,16,0,16)
print("Sleeping for 10 seconds")
import time
time.sleep(10)
print("Stopping file recording")
comedi.comedi_dio_bitfield2(com, 0, 16, 16, 16)
