import struct
import threading
import cStringIO

def _xsum(msg):
    return chr(sum([ord(c) for c in msg]) % 256)

class _parse_num(object):
    types = {1:'<B', 2:'<H', 4:'<I'}
    def __init__(self, length=2, mult=1, unit=""):
        self.t = self.types[length]
        self.m = mult
        self.u = unit
    def __getitem__(self, msg):
        if len(msg) == 3:
            msg += "\x00"
        i, = struct.unpack(self.t, msg)
        return "%0.2f %s"%(i*self.m, self.u)

def parse_status(msg):
    reward_mode = dict(T="Time mode", V="Volume Mode", C="Count mode")
    sensor_status = dict(E="Enable", D="Disable")
    drain_status = dict(P="PC enabled", S="Switch enabled", X="External enabled", D="Disabled", T="Time out")
    ctrl_status = dict(R="Run", H="halted, input voltage high", L="halted, input voltage low", D="Controller device off-line", M="Halted, message problem", O="override timeout")
    
    msg = cStringIO.StringIO(msg)
    order = [
        (1, "Reward mode", reward_mode),
        (1, "Sensor status", sensor_status),
        (1, "Drain status", drain_status),
        (1, "Control status", ctrl_status),
        (1, "Switch State", None),
        (2, "Reward Frequency", _parse_num(unit="s")),
        (1, "Touches per reward", _parse_num(1)),
        (2, "Programmed time", _parse_num(2, 0.1, "ms")),
        (2, "Programmed volume", _parse_num(2, .001, "ml")),
        (2, "Total rewards", _parse_num()),
        (2, "Total touches", _parse_num()),
        (4, "Total reward time", _parse_num(4, 0.1, "ms")),
        (3, "Total reward volume", _parse_num(4, .001, "ml")),
        (5, "Sensor ID", None),
        (5, "Dispenser ID", None),
        (1, "Firmware version", _parse_num(1, .1))
        ]
    
    output = {}
    for length, name, op in order:
        part = msg.read(length)
        if op is None:
            output[name] = part
        else:
            output[name] = op[part]
    
    return output

class ReadMsg(object):
    messages = dict([
        ("&D",(37, "Data", parse_status)), 
        ("#E", (4, "Error")), 
        ("#A", (4, "Acknowledge")), 
        ("*Z", (8, "Volume Calibration"))
    ])
    def __init__(self, port):
        self.port = port
        self.daemon = True
        
    def read(self):
        header = self.port.read(2)
        print self.messages[header][1]+":"
        
        msg = self.port.read(self.messages[header][0] - 2)
        #assert _xsum(header+msg[-1]) == msg[-1], "Wrong checksum! %s"%msg
        if len(self.messages[header]) > 2:
            output = self.messages[header][-1](msg)
            for k, v in output.items():
                print "    %s: %s"%(k, v)
        else:
            print msg
    
    def _write(self, msg):
        fmsg = msg+_xsum(msg)
        self.port.write(fmsg)
        self.read()
    
    def status(self):
        self._write("@CNSNNN")
    
    def reset(self):
        self._write("@CPSNNN")
        self.read()
        self.read()
        
    def reward(self, time=500, volume=None):
        '''Returns the string used to output a time or volume reward.
        
        Parameters
        ----------
        time : int
            Time in milliseconds to turn on the reward
        volume: int
            volume in microliters
        '''
        assert (volume is None and time is not None) or \
            (volume is not None and time is None)
        time /= .1
        
        self._write(struct.pack('<ccxHxx', '@', 'G', time))

if __name__ == "__main__":
    import serial
    port = serial.Serial("/dev/ttyUSB0", baudrate=38400)
    reward = ReadMsg(port)
    reward.status()
    
