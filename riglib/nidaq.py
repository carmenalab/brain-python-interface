import comedi

class Output(object):
    def __init__(self):
        self.ni = comedi.comedi_open("/dev/comedi0")

    def send(self, data):
        #comedi.comedi_data_write(self.ni, 0, 0, data)
        pass

class DataSink(mp.Process):
    def __init__(self, system, **kwargs):
        self.system = system
        self.kwargs = kwargs
        self.pipe, self._pipe = mp.Pipe()
        self.status = mp.Value('b', 0)

    def start(self):
        print "someone started me... wtf??"
        self.status.value = 1
        super(DataSink, self).start()

    def run(self):
        print "starting sink proc"
        system = self.system(**self.kwargs)
        while self.status.value > 0:
            print "i'm running!"
            data = self._pipe.recv()
            system.send(data)

    def send(self, data):
        if self.status.value > 0:
            self.pipe.send(data)

    def stop(self):
        self.status.value = 0
    def __del__(self):
        self.stop()

class NidaqSink(DataSink):
    def __init__(self):
        try:
            from riglib import nidaq
            super(NidaqSink, self).__init__(nidaq.Output)
            self.start()
        except:
            print "No NiDAQ data"