'''
Code for interacting with the Crist reward system(s)
'''


import glob
import time

import struct
import binascii
import threading
import cStringIO
import traceback
import loc_config

import serial
import time

import numpy

try:
    import traits.api as traits
except:
    import enthought.traits.api as traits

def _xsum(msg):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    chrval = map(lambda x: int(''.join(x), 16), zip(*[iter(binascii.b2a_hex(msg))]*2))
    return chr(sum(chrval) % 256)

class _parse_num(object):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    types = {1:'<B', 2:'<H', 4:'<I'}
    def __init__(self, length=2, mult=1, unit=""):
        self.t = self.types[length]
        self.m = mult
        self.u = unit
    def __getitem__(self, msg):
        if len(msg) == 3:
            msg += "\x00"
        i, = struct.unpack(self.t, msg)
        return i*self.m

class Basic(object):
    ''' Docstring '''
    response_message_length = 7
    def __init__(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.port = serial.Serial(glob.glob("/dev/ttyUSB0")[0], baudrate=38400)
        if loc_config.reward_system_version==1: self.set_beeper_volume(128)
        time.sleep(.5)
        self.reset()

    def _write(self, msg):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        fmsg = msg+_xsum(msg)
        #print "sending %r"%fmsg
        self.port.flushOutput()
        self.port.flushInput()
        self.port.write(fmsg)
        # msg_out = self.port.read()
        
        msg_out = self.port.read(self.port.inWaiting())
        print msg_out
        return msg_out

    def reward(self, length):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        length /= .1
        if loc_config.reward_system_version==0:
            self._write(struct.pack('<ccxHxx', '@', 'G', length))
        elif loc_config.reward_system_version==1:
            self._write(struct.pack('<cccHxxx', '@', 'G', '1', length))
        else:
            raise Exception("Unrecognized reward system version!")
        self.port.read(self.port.inWaiting())
    
    def setup_touch_sensor(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if loc_config.reward_system_version==1: #arc system
            cmd = ['@', 'C', '1' ,'O','%c' % 0b10000000, '%c' % 1, '%c' % 0, 'E']
            stuff = ''.join(cmd)
            self._write(stuff)


    def sensor_reward(self, length):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if loc_config.reward_system_version==1:
            cmd = ['@', 'S',  '%c' % 0x02, '%c' % 0x02, '%c' %10, '%c' %0, '%c' %0]
            stuff = ''.join(cmd)
            self._write(stuff)

    def set_beeper_volume(self, volume):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if not (volume >= 0 and volume <= 255):
            raise ValueError("Invalid beeper volume: %g" % volume)
        return self._write('@CS' + '%c' % volume + 'E' + struct.pack('xxx')) 

    def reset(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if loc_config.reward_system_version==0:
            self._write("@CPSNNN")
        elif loc_config.reward_system_version==1:
            cmd = ['@', 'C', '1', 'P', '%c' % 0b10000000, '%c' % 0, '%c' % 0, 'D']
            stuff = ''.join(cmd)
            self._write(stuff)
        else:
            raise Exception("Unrecognized reward system version!")
        self.last_response = self.port.read(self.port.inWaiting())

    def drain(self, drain_time=1200):
        '''
        Turns on the reward system drain for specified amount of time (in seconds)

        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        assert drain_time > 0
        assert drain_time < 9999
        if loc_config.reward_system_version==0: #have to wait and manually tell it to turn off
            self._write("@CNSENN")
            time.sleep(drain_time)
            self._write("@CNSDNN")
        elif loc_config.reward_system_version==1:
            self._write('@M1' + struct.pack('H', drain_time) + 'D' + struct.pack('xx'))
        else:
            raise Exception("Unrecognized reward system version!")

    def drain_off(self):
        '''
        Turns off drain if currently on

        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if loc_config.reward_system_version==0:
            self._write("@CNSDNN")
        elif loc_config.reward_system_version==1:
            self._write('@M1' + struct.pack('H', 0) + 'A' + struct.pack('xx'))
        else:
            raise Exception("Unrecognized reward system version!")

class System(traits.HasTraits, threading.Thread):
    _running = True
    port = traits.Instance(serial.Serial)

    reward_mode = traits.Enum("Time", "Volume", "Count")
    sensor_status = traits.Bool
    drain_status = traits.Enum("Disabled", "PC enabled", "Switch enabled", "External enabled", "Time out")
    ctrl_status = traits.Enum("Run", "Halted high", "Halted low", "Offline", "Halted problem", "Override timeout")
    switch_state = traits.Enum("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F")
    reward_freq = traits.Float
    touches_per_reward = traits.Int
    programmed_time = traits.Float
    programmed_volume = traits.Float
    total_rewards = traits.Int
    total_touches = traits.Int
    total_reward_time = traits.Float
    total_reward_volume = traits.Float
    sensor_id = traits.Str
    dispenser_id = traits.Str
    firmware_version = traits.Float

    def __init__(self, **kwargs):
        super(System, self).__init__(**kwargs)
        threading.Thread.__init__(self)
        self.plock = threading.Lock()
        self.daemon = True
        self._messages = dict([
            ("&D",(37, "Data", self._parse_status)), 
            ("#E", (4, "Error")), 
            ("#A", (4, "Acknowledge")), 
            ("*Z", (8, "Volume Calibration"))
        ])
        reward_mode = dict(T="Time", V="Volume", C="Count")
        drain_status = dict(
            P="PC enabled", 
            S="Switch enabled", 
            X="External enabled", 
            D="Disabled", 
            T="Time out")
        ctrl_status = dict(
            R="Run", 
            H="halted, input voltage high", 
            L="halted, input voltage low", 
            D="Controller device off-line", 
            M="Halted, message problem",
            O="override timeout")
        enable_disable = dict(
            D=False,
            E=True)
        self._order = [
            (1, "reward_mode", reward_mode),
            (1, "sensor_status", enable_disable),
            (1, "drain_status", drain_status),
            (1, "ctrl_status", ctrl_status),
            (1, "switch_state", None),
            (2, "reward_freq", _parse_num(unit="s")),
            (1, "touches_per_reward", _parse_num(1)),
            (2, "programmed_time", _parse_num(2, 0.1, "ms")),
            (2, "programmed_volume", _parse_num(2, .001, "ml")),
            (2, "total_rewards", _parse_num()),
            (2, "total_touches", _parse_num()),
            (4, "total_reward_time", _parse_num(4, 0.1, "ms")),
            (3, "total_reward_volume", _parse_num(4, .001, "ml")),
            (5, "sensor_ID", None),
            (5, "dispenser_ID", None),
            (1, "firmware_version", _parse_num(1, .1))
            ]
        #self.reset_stats()

    def _parse_status(self, msg):
        msg = cStringIO.StringIO(msg)
        output = {}
        for length, name, op in self._order:
            part = msg.read(length)
            if op is None:
                output[name] = part
            else:
                output[name] = op[part]
        
        self.set(**output)

    def __del__(self):
        self._running = False
    
    def run(self):
        while self._running:
            header = self.port.read(2)
            print "recieved %r"%header
            try:
                self.plock.acquire()
                msg = self.port.read(self._messages[header][0] - 2)
                self.plock.release()
                assert _xsum(header+msg[:-1]) == msg[-1], "Wrong checksum! %s"%msg
                if len(self._messages[header]) > 2:
                    self._messages[header][-1](msg)
                else:
                    print self._messages[header], repr(msg)
            except:
                traceback.print_exc()
                time.sleep(10)
                print repr(msg),repr(self.port.read(self.port.inWaiting()))
    
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
    
    def _write(self, msg):
        fmsg = msg+_xsum(msg)
        self.plock.acquire()
        self.port.write(fmsg)
        self.plock.release()
    
    def update(self):
        self._write("@CNSNNN")
    
    def reset(self):
        self._write("@CPSNNN")
    
    def reset_stats(self):
        self._write("@CRSNNN")
    
    def drain(self, status=None):
        mode = ("D", "E")[self.drain_status == "Disabled" if status is None else status]
        self._write("@CNS%sNN"%mode)

def open():
    try:
        #port = serial.Serial(glob.glob("/dev/ttyUSB*")[0], baudrate=38400)
        #reward = System(port=port)
        #reward.start()
        reward = Basic()
        return reward
    except:
        print "Reward system not found"
        import traceback
        import os
        import __builtin__
        traceback.print_exc(file=__builtin__.open(os.path.expanduser('~/code/bmi3d/log/reward.log'), 'w'))
