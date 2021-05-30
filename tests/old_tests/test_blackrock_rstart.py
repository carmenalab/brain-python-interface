from ..features.arduino_features import BlackrockSerialDIORowByte, SerialDIORowByte
from ..riglib import experiment

class par(object):
    def init(self):
        pass

class F(BlackrockSerialDIORowByte, par):
    pass

f = F()
f.init()

