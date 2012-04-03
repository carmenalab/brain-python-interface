import comedi

class Output(object):
    def __init__(self):
        self.ni = comedi.comedi_open("/dev/comedi0")

    def send(self, data):
        #comedi.comedi_data_write(self.ni, 0, 0, data)
        pass