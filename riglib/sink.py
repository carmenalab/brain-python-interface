'''
Generic data sink. Sinks run in separate processes and interact with the main process through code here
'''


import inspect
import traceback
import multiprocessing as mp

import source
from . import FuncProxy

class DataSink(mp.Process):
    def __init__(self, output, **kwargs):
        super(DataSink, self).__init__()
        self.output = output
        self.kwargs = kwargs
        self.cmd_event = mp.Event()
        self.cmd_pipe, self._cmd_pipe = mp.Pipe()
        self.pipe, self._pipe = mp.Pipe()
        self.status = mp.Value('b', 1)
        self.methods = set(n for n in dir(output) if inspect.ismethod(getattr(output, n)))
    
    def run(self):
        output = self.output(**self.kwargs)

        while self.status.value > 0:
            if self._pipe.poll(.001):
                system, data = self._pipe.recv()
                output.send(system, data)

            if self.cmd_event.is_set():
                cmd, args, kwargs = self._cmd_pipe.recv()
                try:
                    if cmd == "getattr":
                        ret = getattr(output, args[0])
                    else:
                        ret = getattr(output, cmd)(*args, **kwargs)
                        
                except Exception as e:
                    traceback.print_exc(file=open('/home/helene/code/bmi3d/log/data_sink_log', 'a'))
                    ret = e
                self._cmd_pipe.send(ret)
                self.cmd_event.clear()
        
        output.close()
        print "ended datasink"
    
    def __getattr__(self, attr):
        if attr in self.methods:
            return FuncProxy(attr, self.cmd_pipe, self.cmd_event)
        else:
            super(DataSink, self).__getattr__(self, attr)

    def send(self, system, data):
        if self.status.value > 0:
            self.pipe.send((system, data))

    def stop(self):
        self.status.value = 0

    def __del__(self):
        self.stop()

class SinkManager(object):
    def __init__(self):
        self.sinks = []
        self.sources = []
        self.registrations = dict()

    def start(self, output, **kwargs):
        print "sinkmanager start %s"%output
        sink = DataSink(output, **kwargs)
        sink.start()
        self.registrations[sink] = set()
        for source, dtype in self.sources:
            sink.register(source, dtype)
            self.registrations[sink].add((source, dtype))
        
        self.sinks.append(sink)
        return sink

    def register(self, system, dtype=None):
        if isinstance(system, source.DataSource):
            name = system.name
            dtype = system.source.dtype
        elif isinstance(system, str):
            name = system
        else:
            name = system.__module__.split(".")[1]
            dtype = system.dtype

        self.sources.append((name, dtype))

        for s in self.sinks:
            if (name, dtype) not in self.registrations[s]:
                self.registrations[s].add((name, dtype))
                s.register(name, dtype)
                
    def send(self, system, data):
        for s in self.sinks:
            s.send(system, data)
    
    def stop(self):
        for s in self.sinks:
            s.stop()

    def __iter__(self):
        for s in self.sinks:
            yield s

#Data Sink manager singleton to be used by features
sinks = SinkManager()


class PrintSink(object):
    '''A null sink which directly prints the received data'''
    def __init__(self):
        print "Starting print sink"
    
    def register(self, name, dtype):
        print "Registered name %s with dtype %r"%(name, dtype)
    
    def send(self, system, data):
        print "Received %s data: \n%r"%(system, data)
    
    def sendMsg(self, msg):
        print "### MESSAGE: %s"%msg
    
    def close(self):
        print "Ended print sink"
