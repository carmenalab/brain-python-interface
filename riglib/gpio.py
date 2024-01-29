'''
Classes for general purpose input output (GPIO)
'''

import types
import numpy as np
#from multiprocessing import Process, Lock
from threading import Thread, Lock, Event
import time
import pyfirmata
from serial.serialutil import SerialException
import serial

from .source import DataSourceSystem

def convert_masked_data_to_pins(mask, data, bits=64):
    ''' Helper to take a mask and some data and turn it into a list of pins and values'''
    pins = []
    values = []
    for pin in range(bits):
        pin_mask = 1 << pin
        if pin_mask & mask:
            pins.append(pin)
            values.append(pin_mask & data > 0)
    return pins, values

class GPIO(object):
    ''' Wrapper for digital i/o'''

    def write(self, pin: int, value: bool):
        pass

    def read(self, pin: int) -> bool:
        pass

    def write_many(self, mask: int, data: int):
        '''
        Write data
        
        mask: sets the address to write
        data: value to write
        '''
        pass

    def read_many(self, mask: int) -> int:
        pass

class TestGPIO(GPIO):

    def __init__(self, pins=14):
        self.value = np.zeros((pins,100))

    def write(self, pin, value):
        self.value[pin,:-1,] = self.value[pin, 1:]
        self.value[pin,-1] = value
        print(".", end="")

    def write_many(self, mask, data):
        pins, values = convert_masked_data_to_pins(mask, data)
        for idx in range(len(pins)):
            self.write(pins[idx], values[idx])

    def read(self, pin):
        return self.value[pin,-1]

class CustomBoard(pyfirmata.Board):

    default_layout = pyfirmata.boards.BOARDS['arduino']
    wait_time = 1

    def __init__(self, port, layout=None, baudrate=112500, name=None, timeout=None):
        self.sp = serial.Serial(port, baudrate, timeout=timeout)
        self.pass_time(self.wait_time)
        self.name = name
        self._layout = layout
        if not self.name:
            self.name = port

        if layout:
            self.setup_layout(layout)
        else:
            layout = self.default_layout
            self.setup_layout(layout)

        # Iterate over the first messages to get firmware data
        while self.bytes_available():
            self.iterate()

class ArduinoGPIO(GPIO):
    ''' Pin-addressable arduino serial interface'''
    def __init__(self, port=None, baudrate=57600, timeout=10, enable_analog=False):
        if port is None:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for p in ports:
                if 'USB' in p.description or 'ACM' in p.description:
                    port = p.device
            if port is None:
                raise Exception('No serial device found')
        self.board = CustomBoard(port, baudrate=baudrate, timeout=timeout)
        self.lock = Lock()
        if enable_analog:
            it = pyfirmata.util.Iterator(self.board)
            it.start()
        self.enable_analog = enable_analog

    def write(self, pin, value):
        with self.lock:
            self.board.digital[pin].write(int(value))
    
    def read(self, pin):
        with self.lock:
            return bool(self.board.digital[pin].read())

    def analog_read(self, pin):
        if not self.enable_analog:
            raise ValueError("Analog reporting not enabled, start over with enable_analog=True!")
        with self.lock:
            if self.board.analog[pin].reporting == False:
                self.board.analog[pin].enable_reporting() # This analog_read() call will fail
            return self.board.analog[pin].read()
        
    def write_many(self, mask, data):
        pins, values = convert_masked_data_to_pins(mask, data)
        for idx in range(len(pins)):
            self.board.digital[pins[idx]].write(values[idx])
        # for p in self.board.digital_ports:
        #     p.write()

    def close(self):
        ''' Call this method before destroying the object'''
        self.board.exit()

class TeensyBoard(CustomBoard):

    default_layout = {
        'digital': tuple(x for x in range(63)),
        'pwm': tuple(x for x in range(63)),
        'analog': tuple(x for x in range(43, 69)),
        'disabled': (0,1),
    }
    wait_time = 0.5 # Teensy is much faster than arduino with hardware Serial port, so we 
                    # don't have to wait so long.


class TeensyGPIO(ArduinoGPIO):

    def __init__(self, port=None, baudrate=112500, timeout=10):
        if port is None:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for p in ports:
                if 'USB Serial' in p.description:
                    port = p.device
            if port is None:
                raise Exception('No serial device found')
        self.board = TeensyBoard(port, baudrate=baudrate, timeout=timeout)
        self.lock = Lock()

    def analog_write(self, pin, value):
        '''
        Write to any pin as a PWM (for digital pins) or analog output (if it supports DAC).
        Can fail if no connection to the board.
        '''
        with self.lock:
            sysex_cmd = 0x6F # extended analog
            data = bytearray([pin, value % 128, value >> 7]) # [pin, lsb, msb]
            try:
                self.board.send_sysex(sysex_cmd, data)
            except ValueError:
                # This happens in BCI experiment sometimes, not sure why. Ignore ~LRS
                pass 
            except SerialException:
                # This happens if the board becomes unplugged, but crashes the game if uncaught.
                raise OSError("Teensy board has become unplugged!")

class NIGPIO(GPIO):

    def __init__(self):
        from riglib.dio.NIUSB6501.py_comedi_control import init_comedi, write_to_comedi
        self.lock = Lock()
        init_comedi()
        self._write_to_comedi = write_to_comedi

    def write(self, pin, value):
        self.write_many(1, value, pin)

    def write_many(self, mask, data, base_channel=0):
        with self.lock:
            ret = self._write_to_comedi(data, mask=mask, base_channel=base_channel)
            if ret == b'':
                raise IOError("Unable to send NIDAQ sync event")

class DigitalWave(Thread):
    ''' Logic-level waves generated by a single or multiple GPIO pins'''

    def __init__(self, gpio, mask=1<<8, data=None):
        ''' Can input a single gpio or multiple'''
        self.gpio = gpio
        self.mask = mask
        if data is None:
            self.data = mask
        else:
            self.data = data
        self.edges = np.zeros(0)
        self.first_edge = True # Note, this doesn't affect the previous laser state
        self._stop_event = Event()
        Thread.__init__(self)
    
    def set_edges(self, edges, first_edge):
        ''' Directly set the list of edges'''
        self.edges = edges
        self.first_edge = first_edge

    def set_pulse(self, duration, first_edge):
        ''' Set the wave to a pulse'''
        self.set_edges([0, duration], first_edge)

    def set_square_wave(self, freq, duration, duty_cycle=0.5, phase_delay=0., first_edge=True):
        ''' Set the wave to a square wave'''
        edges = DigitalWave.square_wave(freq, duration, duty_cycle, phase_delay)
        self.set_edges(edges, first_edge)

    def run(self):
        ''' Generate square pulses defined by edges (in seconds)'''
        t0 = time.perf_counter()
        state = self.first_edge 
        for edge in self.edges:
            while (time.perf_counter() - t0 < edge):
                time.sleep(0)
                if self._stop_event.is_set(): # cancel the wave
                    if state is True:
                        return
                    edge = 0
                    self.edges = []
                    state = False
                    break
            self.gpio.write_many(self.mask, int(state*self.data))
            state = not state

    def stop(self):
        self._stop_event.set()

    @staticmethod
    def delays_to_edges(delays):
        edges = np.zeros(len(delays))
        t = 0
        for i in range(0,len(delays)):
            t += delays[i]
            edges[i] = t
        return edges

    @staticmethod
    def square_wave(freq, duration, duty_cycle=0.5, phase_delay=0):
        '''
        Generate edges for a square wave

        Arguments
        ---------
        freq - frequency of the wave
        duration - total length of the wave, in seconds
        duty_cycle - fraction of each cycle which should be in the ON state
        phase_delay - time in seconds before the start of the wave

        Returns
        -------
        edges - list of edge transition times, in seconds
        '''
        pulse_interval = 1.0/freq
        edge_interval = [pulse_interval*duty_cycle, pulse_interval*(1-duty_cycle)]
        n_edges = int(duration/pulse_interval)
        on_edges = edge_interval[0]*np.ones(n_edges)
        off_edges = edge_interval[1]*np.ones(n_edges)
        delays = np.insert(np.ravel([on_edges, off_edges], order='F'), 0, phase_delay)
        # if len(delays) % 2 == 1:
        #     delays = np.append(delays, edge_interval[1]) # always have an even number of edges
        return DigitalWave.delays_to_edges(delays)

class DigitalWaveMulti(DigitalWave):

    def __init__(self, gpio_list, mask, data):
        ''' Multiple GPIO or lasers'''
        self.edges = np.zeros(0)
        self.first_edge = True # Note, this doesn't affect the previous laser state
        self.gpio = gpio_list
        self.mask = mask
        self.data = data
        Process.__init__(self)
    
    def run(self):
        ''' Generate square pulses defined by edges (in seconds)'''
        t0 = time.perf_counter()
        state = self.first_edge 
        for edge in self.edges:
            while (time.perf_counter() - t0 < edge):
                time.sleep(0)
            for idx in range(len(self.gpio)):
                self.gpio[idx].write_many(self.mask, int(state*self.data))
            state = not state
