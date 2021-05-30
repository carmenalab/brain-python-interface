'''
Code for interacting with the Crist reward system(s). Consult the Crist manual for the command protocol
'''
import time

import struct
import binascii
import threading
import io
import traceback


import serial
import time


try:
    import traits.api as traits
except:
    import enthought.traits.api as traits

def _xsum(msg):
    '''
    Compute the checksums for the messages which must be sent as part of the packet

    Parameters
    ----------
    msg : string
        Message to be sent over the serial port

    Returns
    -------
    char
        The 8-bit checksum of the entire message
    '''
    chrval = [int(''.join(x), 16) for x in zip(*[iter(binascii.b2a_hex(msg))]*2)]
    return chr(sum(chrval) % 256)

class Basic(object):
    '''
    Bare-bones interface for the Crist reward system. Can give timed reward and drain on/off.
    This class is sufficient for all the tasks implemented as of Aug. 2014.
    '''
    response_message_length = 7
    def __init__(self):
        '''
        Constructor for basic reward system interface

        Parameters
        ----------
        None

        Returns
        -------
        Basic instance
        '''
        self.port = serial.Serial('/dev/crist_reward', baudrate=38400)
        from config import config
        self.version = int(config.reward_sys['version'])
        if self.version==1: self.set_beeper_volume(128)
        time.sleep(.5)
        self.reset()

    def _write(self, msg):
        '''
        Send an arbitrary message over the serial port

        Parameters
        ----------
        msg : string
            Message to be sent over the serial port        

        Returns
        -------
        msg_out : string
            Response from crist system after sending command
        '''
        fmsg = msg+_xsum(msg)
        self.port.flushOutput()
        self.port.flushInput()
        self.port.write(fmsg)
        msg_out = self.port.read(self.port.inWaiting())
        return msg_out

    def reward(self, length):
        '''
        Open the solenoid for some length of time

        Parameters
        ----------
        length : float
            Duration of time the solenoid should be open, in seconds. NOTE: in some versions of the system, there appears to be max of ~5s

        Returns
        -------
        None
        '''
        length /= .1
        length = int(length)
        if self.version==0:
            self._write(struct.pack('<ccxHxx', '@', 'G', length))
        elif self.version==1:
            self._write(struct.pack('<cccHxxx', '@', 'G', '1', length))
        else:
            raise Exception("Unrecognized reward system version!")
        self.port.read(self.port.inWaiting())
    
    def setup_touch_sensor(self):
        '''
        Send the serial command to initialize the Crist touch sensor
        '''
        if self.version==1: #arc system
            cmd = ['@', 'C', '1' ,'O','%c' % 0b10000000, '%c' % 1, '%c' % 0, 'E']
            stuff = ''.join(cmd)
            self._write(stuff)

    def sensor_reward(self, length):
        '''
        Set the duration of the reward if the subject touches the touch sensor

        Parameters
        ----------
        length : float
            Duration of time the solenoid should be open, in seconds. NOTE: in some versions of the system, there appears to be max of ~5s

        Returns
        -------
        None
        '''
        if self.version==1:
            cmd = ['@', 'S',  '%c' % 0x02, '%c' % 0x02, '%c' %10, '%c' %0, '%c' %0]
            stuff = ''.join(cmd)
            self._write(stuff)

    def set_beeper_volume(self, volume):
        '''
        Send a command to set the sound level of the audio beep paired with the solenoid opening

        Parameters
        ----------
        volume : int in range [0, 255]
            255 is max possible volume

        Returns
        -------
        string 
            Response message from system
        '''
        if not (volume >= 0 and volume <= 255):
            raise ValueError("Invalid beeper volume: %g" % volume)
        return self._write('@CS' + '%c' % volume + 'E' + struct.pack('xxx')) 

    def reset(self):
        '''
        Send the system reset command
        '''
        if self.version==0:
            self._write("@CPSNNN")
        elif self.version==1:
            cmd = ['@', 'C', '1', 'P', '%c' % 0b10000000, '%c' % 0, '%c' % 0, 'D']
            stuff = ''.join(cmd)
            self._write(stuff)
        else:
            raise Exception("Unrecognized reward system version!")
        self.last_response = self.port.read(self.port.inWaiting())

    def drain(self, drain_time=1200):
        '''
        Turns on the reward system drain for specified amount of time (in seconds)

        Parameters
        ----------
        drain_time : float 
            Time to drain the system, in seconds.

        Returns
        -------
        None
        '''
        assert drain_time > 0
        assert drain_time < 9999
        if self.version == 0: #have to wait and manually tell it to turn off
            self._write("@CNSENN")
            time.sleep(drain_time)
            self._write("@CNSDNN")
        elif self.version == 1:
            self._write('@M1' + struct.pack('H', drain_time) + 'D' + struct.pack('xx'))
        else:
            raise Exception("Unrecognized reward system version!")

    def drain_off(self):
        '''
        Turns off drain if currently on
        '''
        if self.version==0:
            self._write("@CNSDNN")
        elif self.version==1:
            self._write('@M1' + struct.pack('H', 0) + 'A' + struct.pack('xx'))
        else:
            raise Exception("Unrecognized reward system version!")


##########################################
##### Code below this line is unused #####
##########################################
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
        return i*self.m

class System(traits.HasTraits, threading.Thread):
    '''
    More complete reward system interface. Only tested for "version 0" of the system
    '''
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
        msg = io.StringIO(msg)
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
            print("recieved %r"%header)
            try:
                self.plock.acquire()
                msg = self.port.read(self._messages[header][0] - 2)
                self.plock.release()
                assert _xsum(header+msg[:-1]) == msg[-1], "Wrong checksum! %s"%msg
                if len(self._messages[header]) > 2:
                    self._messages[header][-1](msg)
                else:
                    print(self._messages[header], repr(msg))
            except:
                traceback.print_exc()
                time.sleep(10)
                print(repr(msg),repr(self.port.read(self.port.inWaiting())))
    
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
        print("Reward system not found")
        import traceback
        import os
        import builtins
        traceback.print_exc(file=builtins.open(os.path.expanduser('~/code/bmi3d/log/reward.log'), 'w'))
