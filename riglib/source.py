import os
import sys
import time
import inspect
import traceback
import multiprocessing as mp
from multiprocessing import sharedctypes as shm

import numpy as np

import sink
from . import FuncProxy

class DataSource(mp.Process):
    def __init__(self, source, bufferlen=10, **kwargs):
        super(DataSource, self).__init__()
        self.name = source.__module__.split('.')[-1]
        self.filter = None
        self.source = source
        self.source_kwargs = kwargs
        self.bufferlen = bufferlen
        self.max_len = bufferlen*self.source.update_freq
        self.slice_size = self.source.dtype.itemsize
        
        self.lock = mp.Lock()
        self.idx = shm.RawValue('l', 0)
        self.data = shm.RawArray('c', self.max_len*self.slice_size)
        self.pipe, self._pipe = mp.Pipe()
        self.cmd_event = mp.Event()
        self.status = mp.Value('b', 1)
        self.stream = mp.Event()
        self.last_idx = 0

        self.methods = set(n for n in dir(source) if inspect.ismethod(getattr(source, n)))

    def start(self, *args, **kwargs):
        self.sinks = sink.sinks
        super(DataSource, self).start(*args, **kwargs)

    def run(self):
        print "Starting datasource %r"%self.source
        system = self.source(**self.source_kwargs)
        system.start()
        streaming = True
        size = self.slice_size
        while self.status.value > 0:
            if self.cmd_event.is_set():
                cmd, args, kwargs = self._pipe.recv()
                self.lock.acquire()
                try:
                    if cmd == "getattr":
                        ret = getattr(system, args[0])
                    else:
                        ret = getattr(system, cmd)(*args, **kwargs)
                except Exception as e:
                    traceback.print_exc()
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
                self.sinks.send(self.name, data)

                if data is not None:
                    try:
                        self.lock.acquire()
                        i = self.idx.value % self.max_len
                        self.data[i*size:(i+1)*size] = np.array(data).tostring()
                        self.idx.value += 1
                        self.lock.release()
                    except Exception as e:
                        print e
            else:
                time.sleep(.001)
        
        system.stop()
        print "ended datasource %r"%self.source

    def get(self, all=False):
        self.lock.acquire()
        i = (self.idx.value % self.max_len) * self.slice_size
        if all:
            if self.idx.value < self.max_len:
                data = self.data[:i]
            else:
                data = self.data[i:]+self.data[:i]
        else:
            mlen = min((self.idx.value - self.last_idx), self.max_len)
            last = ((self.idx.value - mlen) % self.max_len) * self.slice_size
            if last > i:
                data = self.data[last:] + self.data[:i]
            else:
                data = self.data[last:i]
            
        self.last_idx = self.idx.value
        self.lock.release()
        try:
            data = np.fromstring(data, dtype=self.source.dtype)
        except:
            print "can't get fromstring..."

        if self.filter is not None:
            return self.filter(data)
        return data

    def pause(self):
        self.stream.set()

    def stop(self):
        self.status.value = -1
    
    def __del__(self):
        self.stop()

    def __getattr__(self, attr):
        if attr in self.methods:
            return FuncProxy(attr, self.pipe, self.cmd_event)
        elif not attr.beginsWith("__"):
            print "getting attribute %s"%attr
            self.pipe.send(("getattr", (attr,), {}))
            self.cmd_event.set()
            return self.pipe.recv()
        raise AttributeError(attr)

if __name__ == "__main__":
    from riglib import motiontracker
    sim = DataSource(motiontracker.make_simulate(8))
    sim.start()
    #sim.get()
