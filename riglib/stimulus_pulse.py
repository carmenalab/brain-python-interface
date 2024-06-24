import os
import time
import comedi


class stimulus_pulse(object):
    com = comedi.comedi_open('/dev/comedi0')

    def __init__(self, *args, **kwargs):
        #self.com = comedi.comedi_open('/dev/comedi0')
        super(stimulus_pulse, self).__init__(*args, **kwargs)
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)

        def pulse(self,ts):
            #super(stimulus_pulse, self).pulse()
            subdevice = 0
            write_mask = 0x800000
            val = 0x000000
            base_channel = 0
            while ts < 0.4:
                val = 0x800000
                comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
            else:
                val = 0x000000
                comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
