'''
Generic data sink. Sinks run in separate processes and interact with the main process through code here
'''
from .mp_proxy import FuncProxy, RPCProcess
from . import singleton


class DataSink(RPCProcess):
    '''Generic single-channel data sink'''
    def loop_task(self):
        if self.data_pipe.poll(0.001):
            system, data = self.data_pipe.recv()
            self.target.send(system, data)

    def target_destr(self, ret_status, msg):
        self.target.close()
        print("ended datasink")

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
            self.data_proxy.pipe.send((system, data))


class SinkManager(singleton.Singleton):
    __instance = None
    ''' Data Sink manager singleton to be used by features '''
    def __init__(self):
        '''Don't call this constructor directly. Use SinkManager.get_instance()'''
        super().__init__()
        self.reset()

    def reset(self):
        self.sinks = []
        self.sources = []
        self.registrations = dict()

    def start(self, output, log_filename='', **kwargs):
        '''
        Create a new sink and register with it all the known sources.

        Parameters
        ----------
        output : type
            Data sink target class
        kwargs : optional kwargs
            arguments passed to the data sink target

        Returns
        -------
        sink : DataSink
            Newly-created data sink
        '''
        # print(("sinkmanager start %s"%output))
        sink = DataSink(target_class=output, target_kwargs=kwargs, log_filename=log_filename)
        sink.start()
        self.registrations[sink] = set()
        for source, dtype in self.sources:
            sink.register(source, dtype)
            self.registrations[sink].add((source, dtype))

        self.sinks.append(sink)
        return sink

    def register(self, system, dtype=None, **kwargs):
        '''
        Register a source system with all the known sinks

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
        if hasattr(system, 'name'):
            name = system.name
        elif isinstance(system, str):
            name = system
        else:
            # assume that the system is a class
            print("Inferring name", system.__module__.split("."), type(system))
            name = system.__module__.split(".")[-1]

        if hasattr(system, 'send_to_sinks_dtype'): # used by source.MultiChanDataSource
            dtype = system.send_to_sinks_dtype
        elif hasattr(system, 'source'):
            dtype = system.source.dtype

        self.sources.append((name, dtype))

        for s in self.sinks:
            if (name, dtype) not in self.registrations[s]:
                self.registrations[s].add((name, dtype))
                s.register(name, dtype, **kwargs)

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
