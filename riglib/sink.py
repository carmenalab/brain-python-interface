'''
Generic data sink. Sinks run in separate processes and interact with the main process through code here
'''

import os
import inspect
import traceback
import multiprocessing as mp

from . import source
from .mp_proxy import FuncProxy

class DataSink(mp.Process):
    '''
    Generic single-channel data sink
    '''
    def __init__(self, output, **kwargs):
        '''
        Constructor for DataSink
    
        Parameters
        ----------
        output : type
            data sink class to be implemented in the remote process
        kwargs : optional kwargs
            kwargs to instantiate the data sink
    
        Returns
        -------
        DataSink instance
        '''
        super(DataSink, self).__init__()
        self.output = output
        self.kwargs = kwargs
        self.cmd_event = mp.Event()
        self.cmd_pipe, self._cmd_pipe = mp.Pipe()
        self.pipe, self._pipe = mp.Pipe()
        self.status = mp.Value('b', 1) # mp boolean used for terminating the remote process

        self.methods = set(filter(lambda n: inspect.isfunction(getattr(output, n)), dir(output)))
        # python 2 version: inspect.ismethod doesn't work because the object is not instantiated
        # self.methods = set(n for n in dir(output) if inspect.ismethod(getattr(output, n)))
    
    def run(self):
        '''
        Run the sink system in a remote process

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # instantiate the output interface
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
                    traceback.print_exc(file=open(os.path.join(os.path.dirname(__file__), '../log/data_sink_log'), 'a'))
                    ret = e
                self.cmd_event.clear()
                self._cmd_pipe.send(ret)
        
        # close the sink if the status bit has been set to 0
        output.close()
        print("ended datasink")
    
    def __getattr__(self, attr):
        '''
        Get the specified attribute of the sink in the remote process
    
        Parameters
        ----------
        attr : string
            Name of attribute 
    
        Returns
        -------
        object:
            Value of specified named attribute
        '''
        methods = object.__getattribute__(self, "methods")
        if attr in methods:
            return FuncProxy(attr, self.cmd_pipe, self.cmd_event)
        else:
            raise AttributeError("Can't get attribute: %s. Remote methods available: %s" % (attr, str(self.methods)))

    def send(self, system, data):
        '''
        Send data to the sink system running in the remote process
    
        Parameters
        ----------
        system : string
            Name of system (source) from which the data originated 
        data : object
            Arbitrary data. The remote sink should know how to handle the data
    
        Returns
        -------
        None
        '''
        if self.status.value > 0:
            self.pipe.send((system, data))

    def stop(self):
        '''
        Instruct the sink to stop gracefully by setting the 'status' boolean

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.status.value = 0

    def __del__(self):
        '''
        Stop the remote sink when the object is destructed
        '''
        self.stop()


class SinkManager(object):
    ''' Data Sink manager singleton to be used by features '''
    def __init__(self):
        '''
        Constructor for SinkManager

        Parameters
        ----------
        None

        Returns
        -------
        SinkManager instance
        '''
        self.sinks = []
        self.sources = []
        self.registrations = dict()

    def start(self, output, **kwargs):
        '''
        Docstring

        Parameters
        ----------
        output : DATA_TYPE
            ARG_DESCR
        kwargs : optional kwargs
            ARG_DESCR

        Returns
        -------
        '''
        print(("sinkmanager start %s"%output))
        sink = DataSink(output, **kwargs)
        sink.start()
        self.registrations[sink] = set()
        for source, dtype in self.sources:
            sink.register(source, dtype)
            self.registrations[sink].add((source, dtype))
        
        self.sinks.append(sink)
        return sink

    def register(self, system, dtype=None, **kwargs):
        '''
        Register a system with all the known sinks

        Parameters
        ----------
        system : source.DataSource, source.MultiChanDataSource, or string 
            System to register with all the sinks 
        dtype : None (deprecated)
            Even if specified, this is overwritten in the 'else:' condition below

        Returns
        -------
        None
        '''
        if isinstance(system, source.DataSource):
            name = system.name
            dtype = system.source.dtype
        elif isinstance(system, source.MultiChanDataSource):
            name = system.name
            dtype = system.send_to_sinks_dtype
        elif isinstance(system, str):
            name = system
        else: 
            # assume that the system is a class
            name = system.__module__.split(".")[1]
            dtype = system.dtype

        self.sources.append((name, dtype))

        for s in self.sinks:
            if (name, dtype) not in self.registrations[s]:
                self.registrations[s].add((name, dtype))

                # an unpleasant hack to deal with a change in the default arguments for registering an HDF table
                import hdfwriter
                if s.output == hdfwriter.HDFWriter:
                    s.register(name, dtype, **kwargs)
                else:
                    s.register(name, dtype)
                
    def send(self, system, data):
        '''
        Send data from the specified 'system' to all sinks which have been registered

        Parameters
        ----------
        system: string 
            Name of the system sending the data
        data: np.array
            Generic data to be handled by each sink. Can be a record array, e.g., for task data.

        Returns
        -------
        None
        '''
        for s in self.sinks:
            s.send(system, data)
    
    def stop(self):
        '''
        Run the 'stop' method of all the registered sinks
        '''
        for s in self.sinks:
            s.stop()

    def __iter__(self):
        '''
        Returns a python iterator to allow looping over all the 
        registered sinks, e.g., to send them all the same data
        '''
        for s in self.sinks:
            yield s

# Data Sink manager singleton to be used by features
sinks = SinkManager()


class PrintSink(object):
    '''A null sink which directly prints the received data'''
    def __init__(self):
        print("Starting print sink")
    
    def register(self, name, dtype):
        print(("Registered name %s with dtype %r"%(name, dtype)))
    
    def send(self, system, data):
        print(("Received %s data: \n%r"%(system, data)))
    
    def sendMsg(self, msg):
        print(("### MESSAGE: %s"%msg))
    
    def close(self):
        print("Ended print sink")
