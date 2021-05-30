import time
import inspect
import traceback
import multiprocessing as mp
import io

class PipeWrapper(object):
    def __init__(self, pipe=None, log_filename='', cmd_event=None, **kwargs):
        self.pipe = pipe
        self.log_filename = log_filename
        self.cmd_event = cmd_event

    def log_error(self, err, mode='a'):
        if self.log_filename != '':
            traceback.print_exc(None, err)
            with open(self.log_filename, mode) as fp:
                err.seek(0)
                fp.write(err.read())

    def log_str(self, s, mode="a", newline=True):
        if self.log_filename != '':
            if newline and not s.endswith("\n"):
                s += "\n"
            with open(self.log_filename, mode) as fp:
                fp.write(s)


class FuncProxy(object):
    '''
    Interface for calling functions in remote processes.
    '''
    def __init__(self, name, pipe, event, lock, log_filename=''):
        '''
        Constructor for FuncProxy

        Parameters
        ----------
        name : string
            Name of remote function to call
        pipe : mp.Pipe instance
            multiprocessing pipe through which to send data (function name, arguments) and receive the result
        event : mp.Event instance
            A flag to set which is multiprocessing-compatible (visible to both the current and the remote processes)

        Returns
        -------
        FuncProxy instance
        '''
        self.pipe = pipe
        self.name = name
        self.event = event
        self.lock = lock
        self.log_filename = log_filename

    def log_error(self, err, mode='a'):
        if self.log_filename != '':
            traceback.print_exc(None, err)
            with open(self.log_filename, mode) as fp:
                err.seek(0)
                fp.write(err.read())

    def log_str(self, s, mode="a", newline=True):
        if self.log_filename != '':
            if newline and not s.endswith("\n"):
                s += "\n"
            with open(self.log_filename, mode) as fp:
                fp.write(s)

    def __call__(self, *args, **kwargs):
        '''
        Return the result of the remote function call

        Parameters
        ----------
        *args, **kwargs : positional arguments, keyword arguments
            To be passed to the remote function associated when the object was created

        Returns
        -------
        function result
        '''
        self.lock.acquire()
        self.log_str('lock acquired')

        # block until the remote process unsets the event. Acts as a lock
        n_ticks = 0
        while self.event.is_set() and n_ticks < 100:
            n_ticks += 1
            time.sleep(0.1)

        if n_ticks >= 100:
            raise Exception("Out of sync!")

        # print("FuncProxy.__call__", self.name, args, kwargs)

        self.pipe.send((self.name, args, kwargs))
        self.event.set()
        # resp = self.pipe.recv()
        if self.pipe.poll(10):
            resp = self.pipe.recv()
        else:
            raise Exception("FuncProxy: remote object failed to respond")

        self.lock.release()        
        self.log_str('lock released') 
        return resp


class ObjProxy(PipeWrapper):
    def __init__(self, target_class, *args, **kwargs):
        self.target_class = target_class
        is_instance_method = lambda n: inspect.isfunction(getattr(self.target_class, n))
        self.methods = set(filter(is_instance_method, dir(self.target_class)))

        super().__init__(*args, **kwargs)
        self.lock = mp.Lock()

    def __getattr__(self, attr):
        self.log_str("remotely getting attribute: %s\n" % attr)

        methods = object.__getattribute__(self, "methods")
        pipe = object.__getattribute__(self, "pipe")
        cmd_event = object.__getattribute__(self, "cmd_event")
        lock = object.__getattribute__(self, "lock")

        if attr in methods:
            self.log_str("returning function proxy for %s" % attr)
            return FuncProxy(attr, pipe, cmd_event, lock, log_filename=self.log_filename)
        else:
            self.log_str("sending __getattribute__ over pipe")
            fn_getattr = FuncProxy("__getattribute__", pipe, cmd_event, lock, log_filename=self.log_filename)
            return fn_getattr(attr)

    def set(self, attr, value):
        self.log_str("ObjProxy.setattr")
        pipe = object.__getattribute__(self, "pipe")
        cmd_event = object.__getattribute__(self, "cmd_event")        
        lock = object.__getattribute__(self, "lock")
        setattr_fn = FuncProxy('__setattr__', pipe, cmd_event, lock, log_filename=self.log_filename)

        setattr_fn(attr, value)
        self.log_str("Finished setting remote attr %s to %s" % (str(attr), str(value)))

    def terminate(self):
        self.pipe.send(None)


class DataPipe(PipeWrapper):
    pass

def call_from_remote(x):
    return x

def call_from_parent(x):
    return x

class RPCProcess(mp.Process):
    """mp.Process which implements remote procedure call (RPC) through a mp.Pipe object"""
    def __init__(self, target_class=object, target_kwargs=dict(), log_filename='', **kwargs):
        super().__init__()
        self.cmd_pipe = None
        self.data_pipe = None
        self.log_filename = log_filename

        self.target = None
        self.target_class = target_class
        self.target_kwargs = target_kwargs

        self.cmd_event = mp.Event()
        self.status = mp.Value('b', 1) # mp boolean used for terminating the remote process

        self.target_proxy = None
        self.data_proxy = None

    def __getattr__(self, attr):
        """ Redirect attribute access to the target object if the 
        attribute can't be found in the process wrapper """
        try:
            if self.target_proxy is not None and self.status.value > 0:
                try:
                    return getattr(self.target_proxy, attr)
                except:
                    raise AttributeError("RPCProcess: could not forward getattr %s to target of type %s" % (attr, self.target_class))
            else:
                raise AttributeError("RPCProcess: target proxy not initialized")
        except:
            raise AttributeError("Could not get RPCProcess attribute: %s" % attr)

    def log_error(self, err, mode='a'):
        if self.log_filename != '':
            with open(self.log_filename, mode) as fp:
                traceback.print_exc(file=fp)
                fp.write(str(err))

    def log_str(self, s, mode="a", newline=True):
        if self.log_filename != '':
            if newline and not s.endswith("\n"):
                s += "\n"
            with open(self.log_filename, mode) as fp:
                fp.write(s)

    @call_from_remote
    def target_constr(self):
        try:
            self.target = self.target_class(**self.target_kwargs)
            if hasattr(self.target, 'start'):
                self.target.start()
        except Exception as e:
            print("RPCProcess.target_constr: unable to start source!")
            print(e)

            import io
            err = io.StringIO()
            self.log_error(err, mode='a')
            err.seek(0)

            self.status.value = -1

    @call_from_remote
    def target_destr(self, ret_status, msg):
        pass

    @call_from_parent
    def start(self):
        self.cmd_pipe, pipe_end2 = mp.Pipe()
        self.data_pipe, data_pipe2 = mp.Pipe()

        self.target_proxy = ObjProxy(self.target_class, pipe=pipe_end2, 
            log_filename=self.log_filename, cmd_event=self.cmd_event)
        self.data_proxy = DataPipe(data_pipe2)
        super().start()
        return self.target_proxy, self.data_proxy

    def check_run_condition(self):
        return self.status.value > 0

    def is_enabled(self):
        return self.status.value > 0

    @call_from_parent
    def stop(self):
        self.status.value = -1

    def __del__(self):
        '''Stop the process when the object is destructed'''
        if self.status.value > 0:
            #self.stop() <- currently causing issues with task_wrapper. Somewhere the status is not being set properly after the task ends...
            self.status.value = -1

    def is_cmd_present(self):
        return self.cmd_event.is_set()

    def clear_cmd(self):
        self.log_str("clearing")
        self.cmd_event.clear()

    @call_from_remote
    def loop_task(self):
        time.sleep(0.01)

    @call_from_remote
    def run(self):
        self.log_str("RPCProcess.run")

        self.target_constr()

        try:
            while self.is_enabled():
                if not self.check_run_condition():
                    self.log_str("The target's termination condition was reached")
                    break

                if self.is_cmd_present():
                    self.proc_rpc_command()
                    self.clear_cmd()

                self.loop_task()

            self.log_str("RPCProcess.run: end of while loop")
            ret_status, msg = 0, ''
        except KeyboardInterrupt:
            ret_status, msg = 1, 'KeyboardInterrupt'
        except:
            import traceback
            traceback.print_exc()
            err = io.StringIO()
            self.log_error(err, mode='a')
            err.seek(0)
            ret_status, msg = 1, err.read()
        finally:
            self.target_destr(ret_status, msg)
            self.status.value = -1

    @call_from_remote
    def proc_rpc_command(self):
        cmd = self.cmd_pipe.recv()
        self.log_str("Received command: %s" % str(cmd))

        if cmd is None:
            self.stop()
            return 

        try:
            fn_name, cmd_args, cmd_kwargs = cmd
            fn = getattr(self.target, fn_name)
            self.log_str("Function: " + str(fn))

            fn_output = fn(*cmd_args, **cmd_kwargs)
            self.cmd_pipe.send(fn_output)

            self.log_str('Done with command: %s, output=%s\n\n' % (fn_name, fn_output))
        except Exception as e:
            err = io.StringIO()
            self.log_error(err, mode='a')

            self.cmd_pipe.send(e)

