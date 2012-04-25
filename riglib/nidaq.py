import comedi

import shm

class Output(object):
    def __init__(self, systems=None):
        self.ni = comedi.comedi_open("/dev/comedi0")

    def send(self, system, data):
        #comedi.comedi_data_write(self.ni, 0, 0, data)
        pass