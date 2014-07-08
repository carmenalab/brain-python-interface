import comedi
import time

com = comedi.comedi_open("/dev/comedi0")

for i in range(6):
	# set strobe pin low
	comedi.comedi_dio_bitfield2(com, 0, 1, 0, 16)
	time.sleep(1)

	# set strobe pin high
	comedi.comedi_dio_bitfield2(com, 0, 1, 1, 16)
	time.sleep(1)
