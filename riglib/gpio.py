'''
Classes for general purpose input output (GPIO)
'''

import numpy as np
#from multiprocessing import Process, Lock
from threading import Thread as Process, Lock
import time
import pyfirmata

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

class ArduinoGPIO(GPIO):
    ''' Pin-addressable arduino serial interface'''
    def __init__(self, port=None, baudrate=57600, timeout=10):
        if port is None:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for p in ports:
                if 'USB' in p.description or 'ACM' in p.description:
                    port = p.device
            if port is None:
                raise Exception('No serial device found')
        self.board = pyfirmata.Arduino(port, baudrate=baudrate, timeout=timeout)
        self.lock = Lock()

    def write(self, pin, value):
        with self.lock:
            self.board.digital[pin].write(int(value))
    
    def read(self, pin):
        with self.lock:
            return bool(self.board.digital[pin].read())

    def write_many(self, mask, data):
        pins, values = convert_masked_data_to_pins(mask, data)
        for idx in range(len(pins)):
            self.board.digital[pins[idx]].write(values[idx])
        # for p in self.board.digital_ports:
        #     p.write()

    def close(self):
        ''' Call this method before destroying the object'''
        self.board.exit()

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
        teensy = {
            'digital': tuple(x for x in range(63)),
            'pwm': tuple(x for x in range(63)),
            'analog': tuple(x for x in range(43, 69)),
            'disabled': (0,1),
        }
        self.board = pyfirmata.Board(port, baudrate=baudrate, timeout=timeout, layout=teensy)
        self.lock = Lock()

    def analog_write(self, pin, value):
        '''
        Write to any pin as a PWM (for digital pins) or analog output (if it supports DAC)
        '''
        with self.lock:
            msg = [0xF0, 0x6F, pin, value % 128, value >> 7, 0xF7] # [START_SYSEX, extended analog, pin, lsb, msb, END_SYSEX]
            self.board.sp.write(msg)

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

class DigitalWave(Process):
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
        Process.__init__(self)
    
    def set_edges(self, edges, first_edge):
        ''' Directly set the list of edges'''
        self.edges = edges
        self.first_edge = first_edge

    def set_pulse(self, duration, first_edge):
        ''' Set the wave to a pulse'''
        self.set_edges([0, duration], first_edge)

    def set_square_wave(self, freq, duration, first_edge=True):
        ''' Set the wave to a square wave'''
        edges = DigitalWave.square_wave(freq, duration)
        self.set_edges(edges, first_edge)

    def run(self):
        ''' Generate square pulses defined by edges (in seconds)'''
        t0 = time.perf_counter()
        state = self.first_edge 
        for edge in self.edges:
            while (time.perf_counter() - t0 < edge):
                pass
            self.gpio.write_many(self.mask, int(state*self.data))
            state = not state

    @staticmethod
    def delays_to_edges(delays):
        edges = np.zeros(len(delays))
        t = 0
        for i in range(0,len(delays)):
            t += delays[i]
            edges[i] = t
        return edges

    @staticmethod
    def square_wave(freq, duration):
        pulse_interval = 1.0/freq
        edge_interval = pulse_interval/2
        if duration < edge_interval: # less than one half wavelength
            edge_interval = duration
        length = int(duration/edge_interval)
        delays = np.insert(edge_interval*np.ones(length), 0, 0)
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
                pass
            for idx in range(len(self.gpio)):
                self.gpio[idx].write_many(self.mask, int(state*self.data))
            state = not state