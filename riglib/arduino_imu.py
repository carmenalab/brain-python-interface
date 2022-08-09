
import time
import numpy as np

import serial
from .source import DataSourceSystem


class System(DataSourceSystem):
    '''
    Generic DataSourceSystem interface for the Phidgets board: http://www.phidgets.com/products.php?category=0&product_id=1018_2
    '''
    update_freq = 20

    def __init__(self, n_sensors=10, n_inputs=0):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.n_sensors = n_sensors
        self.n_inputs = n_inputs
        self.interval = 1. / self.update_freq

        self.sensordat = np.zeros((n_sensors + 1,))
        self.inputdat = np.zeros((n_inputs,), dtype=np.bool)
        self.data = np.zeros((1,), dtype=self.dtype)
        self.port = serial.Serial('/dev/arduino_joystick', baudrate=115200)
        self.port.flush()
        time.sleep(3.)
        self.t0 = time.time()
    def start(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.tic = time.time()

    def stop(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        pass
    
    def get(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''

        toc = time.time() - self.tic
        if 0 < toc < self.interval:
            time.sleep(self.interval - toc)
        try:
            self.port.write('d')
            #time.sleep(self.interval/20.)
            for i in range(self.n_sensors):
                s = float(self.port.readline())
                self.sensordat[i] = s
            self.sensordat[self.n_sensors] = time.time()
        except:
            print('sensor_error', self.tic - self.t0, self.sensordat, self.n_sensors)
        #self.port.flush()

        self.data['sensors'] = self.sensordat
        self.data['inputs'] = self.inputdat
        self.tic = time.time()
        return self.data
    
    def sendMsg(self, msg):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        pass

    def __del__(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.port.close()

def make(sensors, inputs, cls=System, **kwargs):
    '''
    Docstring
    This ridiculous function dynamically creates a class with a new init function

    Parameters
    ----------

    Returns
    -------
    '''
    def init(self, **kwargs):
        print('making arduino imu')
        super(self.__class__, self).__init__(n_sensors=sensors, n_inputs=inputs, **kwargs)
        print('making arduino imu2')

    dtype = np.dtype([('sensors', np.float, (sensors+1,)), ('inputs', np.bool, (inputs,))])
    return type(cls.__name__, (cls,), dict(dtype=dtype, __init__=init))
