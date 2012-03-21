import os
import inspect
import multiprocessing as mp
from multiprocessing import sharedctypes as shm

import numpy as np

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
        self.bufferlen = bufferlen

        self.lock = mp.Lock()
        self.idx = shm.RawValue('l', 0)
        self.data = shm.RawArray('d', self.max_size*self.slice_size)
        self.pipe, self._pipe = mp.Pipe()
        self.cmd_event = mp.Event()
        self.status = mp.Value('b', 1)
        self.stream = mp.Event()

        self.methods = set(n for n in dir(source) if inspect.ismethod(getattr(source, n)))

    def run(self):
        system = self.source(**self.source_kwargs)
        system.start()
        streaming = True

        size = self.slice_size
        while self.status.value > 0:
            if self.cmd_event.is_set():
                cmd, args, kwargs = self._pipe.recv()
                self.lock.acquire()
                try:
                    ret = getattr(system, cmd)(*args, **kwargs)
                except Exception as e:
                    ret = e
                self.lock.release()
                self._pipe.send(ret)
                self.cmd_event.clear()
            
            if self.stream.is_set():
                self.stream.clear()
                streaming = not streaming
                if streaming:
                    self.idx.value = 0
                    system.start()
                else:
                    system.stop()

            if streaming:
                data = system.get()
                if data is not None:
                    try:
                        self.lock.acquire()
                        i = self.idx.value % self.max_size
                        self.data[i*size:(i+1)*size] = np.ravel(data)
                        self.idx.value += 1
                        self.lock.release()
                    except:
                        print repr(data)
            else:
                time.sleep(.001)

        print "ending data collection"
        system.stop()

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

    def pause(self):
        self.stream.set()

    def __del__(self):
        self.status.value = -1

    def __getattr__(self, attr):
        if attr in self.methods:
            return FuncProxy(attr, self.pipe, self.cmd_event)
        else:
            self.pipe.send(("__getattr__", (attr,), {}))
            self.cmd_event.set()
            return self.pipe.recv()

class EyeData(DataSource):
    def __init__(self, **kwargs):
        from riglib import eyetracker
        super(EyeData, self).__init__(eyetracker.System, **kwargs)
        self.max_size = self.bufferlen*500

class EyeSimulate(DataSource):
    def __init__(self, **kwargs):
        from riglib import eyetracker
        super(EyeSimulate, self).__init__(eyetracker.Simulate, **kwargs)
        self.max_size = self.bufferlen*500

class MotionData(DataSource):
    def __init__(self, marker_count=8, **kwargs):
        from riglib import motiontracker
        self.slice_size = marker_count * 3
        super(MotionData, self).__init__(motiontracker.System, marker_count=marker_count, **kwargs)
        self.max_size = self.bufferlen*480

    def get(self):
        data = super(MotionData, self).get()
        try:
            return data.reshape(len(data), -1, 3)
        except:
            print "Data size wrong! %d"%len(data)
            return np.array([])

class MotionSimulate(DataSource):
    def __init__(self, marker_count = 8, **kwargs):
        from riglib import motiontracker
        self.slice_size = marker_count * 3
        super(MotionSimulate, self).__init__(motiontracker.Simulate, marker_count=marker_count, **kwargs)
        self.max_size = self.bufferlen*480

    def get(self):
        data = super(MotionSimulate, self).get()
        try:
            return data.reshape(len(data), -1, 3)
        except:
            print "Data size wrong! %d"%len(data)
            return np.array([])

if __name__ == "__main__":
    sim = MotionSimulate()
    sim.start()
    #sim.get()
