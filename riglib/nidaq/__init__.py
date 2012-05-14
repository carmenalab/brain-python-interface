import pcidio

class System(object):
    def __init__(self, device="/dev/comedi0"):
        self.systems = dict()
        if pcidio.init(device) != 0:
            raise ValueError("Could not initialize comedi system")
    
    def close(self):
        if pcidio.closeall() != 0:
            raise ValueError("Unable to close comedi system")

    def register(self, system, dtype):
        print "nidaq register %s"%system
        self.systems[system] = pcidio.register_sys(system, str(dtype.descr))

    def sendMsg(self, msg):
        pcidio.sendMsg(str(msg))

    def send(self, system, data):
        s = self.systems[system]
        pcidio.sendData(s, data.tostring())

    def sendRow(self, system, idx):
        s = self.systems[system]
        pcidio.sendRow(s, idx)
