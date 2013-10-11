import time
import itertools
import numpy as np

from Phidgets.Devices.InterfaceKit import InterfaceKit

class System(object):
    update_freq = 1000

    def __init__(self, n_sensors=2, n_inputs=1):
        self.n_sensors = n_sensors
        self.n_inputs = n_inputs
        self.interval = 1. / self.update_freq

        self.sensordat = np.zeros((n_sensors,))
        self.inputdat = np.zeros((n_inputs,), dtype=np.bool)
        self.data = np.zeros((1,), dtype=self.dtype)

        self.kit = InterfaceKit()
        self.kit.openPhidget()
        self.kit.waitForAttach(10)
    
    def start(self):
        self.tic = time.time()

    def stop(self):
        pass
    
    def get(self):
        toc = time.time() - self.tic
        if 0 < toc < self.interval:
            time.sleep(self.interval - toc)
        for i in range(self.n_sensors):
            self.sensordat[i] = self.kit.getSensorValue(i) / 1000.
        for i in range(self.n_inputs):
            self.inputdat[i] = self.kit.getInputState(i)
        self.data['sensors'] = self.sensordat
        self.data['inputs'] = self.inputdat
        self.tic = time.time()
        return self.data
    
    def sendMsg(self, msg):
        pass

    def __del__(self):
        self.kit.closePhidget()

def make(sensors, inputs, cls=System, **kwargs):
    """This ridiculous function dynamically creates a class with a new init function"""
    def init(self, **kwargs):
        super(self.__class__, self).__init__(n_sensors=sensors, n_inputs=inputs, **kwargs)
    
    dtype = np.dtype([('sensors', np.float, (sensors,)), ('inputs', np.bool, (inputs,))])
    return type(cls.__name__, (cls,), dict(dtype=dtype, __init__=init))