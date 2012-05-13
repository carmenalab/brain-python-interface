import pcidio

class System(object):
    def __init__(self, device="/dev/comedi0"):
        pcidio.init(device)
        self.systems = dict()

    def register(self, system, dtype):
        self.systems[system] = pcidio.register_sys(system, dtype)

    def sendMsg(self, msg):
        pcidio.sendMsg(msg)

    def send(self, system, data):
        s = self.systems[system]
        pcidio.sendData(s, data.tostring())