import os
import inspect
import multiprocessing as mp
from multiprocessing import sharedctypes as shm

import numpy as np

from riglib import eyetracker, motiontracker

#Update frequency for each datasource, for calculating size of shm
update_freq = {
    eyetracker.System:500, motiontracker.System:480,
    eyetracker.Simulate:500, motiontracker.Simulate:480
}

class FuncProxy(object):
    def __init__(self, name, pipe, event):
        self.pipe = pipe
        self.name = name
        self.event = event

    def __call__(self, *args, **kwargs):
        self.pipe.send((self.name, args, kwargs))
        self.event.set()
        return self.pipe.recv()

class DataSource(mp.Process):
    slice_size = 2
    def __init__(self, source, bufferlen=10, **kwargs):
        super(DataSource, self).__init__()
        self.filter = None
        self.source = source
        self.source_kwargs = kwargs
        self.max_size = bufferlen*update_freq[source]

        self.lock = mp.Lock()
        self.idx = shm.RawValue('l', 0)
        self.data = shm.RawArray('d', self.max_size*self.slice_size)
        self.pipe, self._pipe = mp.Pipe()
        self.cmd_event = mp.Event()
        self.status = mp.Value('b', 1)

        self.methods = set(n for n in dir(source) if inspect.ismethod(getattr(source, n)))

    def run(self):
        self.system = self.source(**self.source_kwargs)
        self.system.start()

        size = self.slice_size
        while self.status.value > 0:
            if self.cmd_event.is_set():
                cmd, args, kwargs = self._pipe.recv()
                self.lock.acquire()
                try:
                    ret = getattr(self.system, cmd)(*args, **kwargs)
                except Exception as e:
                    ret = e
                self.lock.release()
                self._pipe.send(ret)
                self.cmd_event.clear()

            data = self.system.get()
            if data is not None:
                self.lock.acquire()
                i = self.idx.value % self.max_size
                self.data[i*size:(i+1)*size] = data.ravel()
                self.idx.value += 1
                self.lock.release()

        print "ending data collection"
        self.system.stop()

    def get(self):
        self.lock.acquire()
        i = (self.idx.value % self.max_size) * self.slice_size
        if self.idx.value > self.max_size:
            data = self.data[i:]+self.data[:i]
        else:
            data = self.data[:i]
        self.idx.value = 0
        self.lock.release()
        try:
            data = np.array(data).reshape(-1, self.slice_size)
        except:
            print "can't reshape, len(data)=%d, size[source]=%d"%(len(data), self.slice_size)

        if self.filter is not None:
            return self.filter(data)
        return data

    def stop(self):
        self.status.value = -1

    def __del__(self):
        self.stop()

    def __getattr__(self, attr):
        if attr in self.methods:
            return FuncProxy(attr, self.pipe, self.cmd_event)
        else:
            self.pipe.send(("__getattr__", (attr,), {}))
            self.cmd_event.set()
            return self.pipe.recv()

class EyeData(DataSource):
    def __init__(self, **kwargs):
        super(EyeData, self).__init__(eyetracker.System, **kwargs)

class MotionData(DataSource):
    def __init__(self, marker_count=8, **kwargs):
        self.slice_size = marker_count * 3
        super(MotionData, self).__init__(motiontracker.System, marker_count=marker_count, **kwargs)

    def get(self):
        data = super(MotionData, self).get()
        return data.reshape(len(data), -1, 3)


class EyeSimulate(DataSource):
    def __init__(self, **kwargs):
        super(EyeSimulate, self).__init__(eyetracker.Simulate, **kwargs)

class MotionSimulate(DataSource):
    def __init__(self, marker_count = 8, **kwargs):
        self.slice_size = marker_count * 3
        super(MotionSimulate, self).__init__(motiontracker.Simulate, marker_count=marker_count, **kwargs)

    def get(self):
        data = super(MotionSimulate, self).get()
        return data.reshape(len(data), -1, 3)

if __name__ == "__main__":
    sim = MotionSimulate()
    sim.start()
    #sim.get()