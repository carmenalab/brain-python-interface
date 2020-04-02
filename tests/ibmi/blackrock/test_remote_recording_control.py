import comedi
import time

com = comedi.comedi_open("/dev/comedi0")

# start (or resume) recording
print('starting recording')

# set strobe pin low
comedi.comedi_dio_bitfield2(com, 0, 1, 0, 16)

# set last data pin ("D15"; 16th pin) high
comedi.comedi_dio_bitfield2(com, 0, 1, 1, 15)

# set strobe pin high
comedi.comedi_dio_bitfield2(com, 0, 1, 1, 16)

# set strobe pin low
comedi.comedi_dio_bitfield2(com, 0, 1, 0, 16)


print('waiting for 5 seconds...')
time.sleep(5)

# stop recording
print('stopping recording')

# strobe pin should already be low

# set last data pin ("D15"; 16th pin) low
comedi.comedi_dio_bitfield2(com, 0, 1, 0, 15)

# set strobe pin high
comedi.comedi_dio_bitfield2(com, 0, 1, 1, 16)

# set strobe pin low
comedi.comedi_dio_bitfield2(com, 0, 1, 0, 16)


