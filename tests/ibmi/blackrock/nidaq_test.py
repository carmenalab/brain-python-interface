import comedi
import time

com = comedi.comedi_open("/dev/comedi0")

nchan = comedi.comedi_get_n_channels(com, 0)
print("found comedi system with %d channels" % nchan)

for i in range(nchan):
	comedi.comedi_dio_config(com, 0, i, comedi.COMEDI_OUTPUT)

for i in range(4):
	comedi.comedi_dio_bitfield2(com, 0, 1, 1, 0)
	time.sleep(1)

	comedi.comedi_dio_bitfield2(com, 0, 1, 0, 0)
	time.sleep(1)
