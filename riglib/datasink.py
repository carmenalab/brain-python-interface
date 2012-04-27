import inspect
import traceback
import multiprocessing as mp

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
                        print "getting %s"%repr(args)
                        ret = getattr(output, args[0])
                    else:
                        ret = getattr(output, cmd)(*args, **kwargs)
                        
                except Exception as e:
                    traceback.print_exc()
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

    def start(self, output, **kwargs):
        print "sinkmanager start %s"%output
        sink = DataSink(output, systems=self.sources, **kwargs)
        sink.start()
        self.sinks.append(sink)

    def register(self, system):
        print "Registering a %r system"%system
        self.sources.append(system)

        for s in self.sinks:
            s.register(system)
    
    def stop(self):
        for s in self.sinks:
            s.stop()

    def __iter__(self):
        for s in self.sinks:
            yield s

#Data Sink manager singleton to be used by features
sinks = SinkManager()
