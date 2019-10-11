"""Source template for a serial port streaming sensor, e.g., from an Arduino"""
import serial
import numpy as np
import traceback

from riglib.source import DataSourceSystem

class SerialPortSource(DataSourceSystem):
    default_response = []
    dtype = np.dtype([("sensor_data", np.float)])
    update_freq = 100

    def __init__(self, port, baudrate):
        self.port = serial.Serial(port, baudrate)
        self.port.flushInput()
        self.port.reset_output_buffer()

    def start(self):
        self.port.flushInput()
        self.port.reset_output_buffer()

    def stop(self):
        self.port.flushInput()
        self.port.reset_output_buffer()
        self.port.close()

    def get(self):
        '''
        Retrieve the current data available from the source. 
        '''
        
        data = self.default_response
        try:
            data_raw = self.port.readline().rstrip().decode('utf-8')
            data = np.zeros((1,), dtype=self.dtype)
            data_vals = tuple(float(x) for x in data_raw.split(', '))

            if len(self.dtype) == 1 and len(data_vals) > 1:
                data[self.dtype.names[0]] = data_vals
            elif len(self.dtype) == len(data_vals):
                for name, val in zip(self.dtype.names, data_vals):
                    data[name] = val
        except:
            traceback.print_exc()
            pass
            
        return data

class SerialPortFeature(object):
    pass