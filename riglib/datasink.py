import inspect
import traceback
import multiprocessing as mp
from riglib import FuncProxy

class DataSink(mp.Process):
    def __init__(self, output, **kwargs):
        super(DataSink, self).__init__()
        self.output = output
        self.kwargs = kwargs
        self.cmd_event = mp.Event()
        self.pipe, self._pipe = mp.Pipe()
        self.status = mp.Value('b', 1)
        self.methods = set(n for n in dir(output) if inspect.ismethod(getattr(output, n)))
    
    def run(self):
        print "starting sink proc"
        output = self.output(**self.kwargs)
        while self.status.value > 0:
            if self.cmd_event.is_set():
                cmd, args, kwargs = self._pipe.recv()
                try:
                    if cmd == "getattr":
                        print "getting %s"%repr(args)
                        ret = getattr(system, args[0])
                    else:
                        ret = getattr(system, cmd)(*args, **kwargs)
                except Exception as e:
                    traceback.print_exc()
                    ret = e
                self._pipe.send(ret)
                self.cmd_event.clear()

            if self._pipe.poll(1):
                system, data = self._pipe.recv()
                output.send(system, data)
    
    def __getattr__(self, attr):
        if attr in self.methods:
            return FuncProxy(attr, self.pipe, self.cmd_event)
        else:
            super(DataSink, self).__getattr__(self, attr)

    def send(self, system, data):
        print "sending data %s"%data
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
        print "blah", sink, self.instances

    def register(self, system):
        self.sources.append(system)
        for s in self.sources:
            s.register(system)

    def send(self, system, data):
        for s in self.sinks:
            s.send(system, data)
    
    def stop(self):
        for s in self.sinks:
            s.stop()

sinks = SinkManager()