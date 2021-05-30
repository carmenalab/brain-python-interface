'''
Code for interacting with the Phdigets API
'''
import time
import numpy as np

from Phidgets.Devices.InterfaceKit import InterfaceKit
from .source import DataSourceSystem

class System(DataSourceSystem):
    '''
    Generic DataSourceSystem interface for the Phidgets board: http://www.phidgets.com/products.php?category=0&product_id=1018_2
    '''
    update_freq = 1000

    def __init__(self, n_sensors=2, n_inputs=1):
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

        self.sensordat = np.zeros((n_sensors,))
        self.inputdat = np.zeros((n_inputs,), dtype=np.bool)
        self.data = np.zeros((1,), dtype=self.dtype)

        self.kit = InterfaceKit()
        self.kit.openPhidget()
        self.kit.waitForAttach(2000)
    
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
            for i in range(self.n_sensors):
                self.sensordat[i] = self.kit.getSensorValue(i) / 1000.
            for i in range(self.n_inputs):
                self.inputdat[i] = self.kit.getInputState(i)
        except:
            print('sensor_error')
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
        self.kit.closePhidget()

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
        super(self.__class__, self).__init__(n_sensors=sensors, n_inputs=inputs, **kwargs)
    
    dtype = np.dtype([('sensors', np.float, (sensors,)), ('inputs', np.bool, (inputs,))])
    return type(cls.__name__, (cls,), dict(dtype=dtype, __init__=init))
