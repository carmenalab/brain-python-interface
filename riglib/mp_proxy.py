import os
import sys
import time
import inspect
import traceback
import multiprocessing as mp
from multiprocessing import sharedctypes as shm
import io
import numpy as np


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
    def __init__(self, name, pipe, event=None, lock=None):
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
        if not self.lock is None:
            self.lock.acquire()

        print("FuncProxy.__call__", self.name, args, kwargs)
        self.pipe.send((self.name, args, kwargs))
        if not self.event is None:
            self.event.set()
        resp = self.pipe.recv()

        if not self.lock is None:
            self.lock.release()        
        return resp


class ObjProxy(PipeWrapper):
    def __init__(self, *args, **kwargs):
        self.methods = kwargs.pop('methods', [])
        super().__init__(*args, **kwargs)
        self.lock = mp.Lock()

    def __getattr__(self, attr):
        self.log_str("remotely getting attribute: %s\n" % attr)

        methods = object.__getattribute__(self, "methods")
        pipe = object.__getattribute__(self, "pipe")
        cmd_event = object.__getattribute__(self, "cmd_event")
        lock = object.__getattribute__(self, "lock")
        print("methods:", methods)
        if attr in methods:
            self.log_str("returning function proxy for %s" % attr)
            return FuncProxy(attr, pipe, cmd_event, lock)
        else:
            self.log_str("sending __getattribute__ over pipe")
            fn_getattr = FuncProxy("__getattribute__", pipe, cmd_event, lock)
            return fn_getattr(attr)
            # pipe.send(("__getattribute__", [attr], {}))
            # cmd_event.set()
            # ret = pipe.recv()
            # if isinstance(ret, Exception): 
            #     raise AttributeError("Can't get attribute: %s. Remote methods available: %s" % (attr, str(self.methods)))

            # return ret

    def set(self, attr, value):
        self.log_str("ObjProxy.setattr")
        pipe = object.__getattribute__(self, "pipe")
        cmd_event = object.__getattribute__(self, "cmd_event")        
        lock = object.__getattribute__(self, "lock")
        setattr_fn = FuncProxy('__setattr__', pipe, cmd_event, lock)

        setattr_fn(attr, value)
        self.log_str("Finished setting remote attr %s to %s" % (str(attr), str(value)))

    def terminate(self):
        self.pipe.send(None)


class DataPipe(PipeWrapper):
    pass

class RPCProcess(mp.Process):
    """mp.Process which implements remote procedure call (RPC) through a mp.Pipe object"""
    proxy = ObjProxy
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cmd_pipe = None
        self.data_pipe = None
        self.log_filename = kwargs.pop('log_filename', '')
        if self.log_filename != '':
            with open(self.log_filename, 'w') as f:
                f.write('')
        self.run_the_loop = False
        self._remote_methods = None
        self.target = None

        self.cmd_event = mp.Event()
        self.status = mp.Value('b', 1) # mp boolean used for terminating the remote process

        self.target_proxy = None
        self.data_proxy = None

    @property
    def remote_class(self):
        raise NotImplementedError("Specify the type of self.target to distinguish methods from attributes")

    def remote_methods(self):
        if self._remote_methods is None:
            is_instance_method = lambda n: inspect.isfunction(getattr(self.remote_class, n))
            self._remote_methods = set(filter(is_instance_method, dir(self.remote_class)))
        return self._remote_methods

    def __getattr__(self, attr):
        """ Redirect attribute access to the target object if the 
        attribute can't be found in the process wrapper """
        try:
            if self.target_proxy is not None:
                try:
                    return getattr(self.target_proxy, attr)
                except:
                    raise AttributeError("RPCProcess: could not forward getattr %s to target of type %s" % (attr, self.remote_class))
            else:
                raise AttributeError("RPCProcess: target proxy not initialized")
        except:
            raise AttributeError("Could not get RPCProcess attribute: %s" % attr)

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

    def target_constr(self):
        pass

    def target_destr(self, ret_status, msg):
        pass

    def start(self):
        self.cmd_pipe, pipe_end2 = mp.Pipe()
        self.data_pipe, data_pipe2 = mp.Pipe()

        self.target_proxy = self.proxy(pipe=pipe_end2, log_filename=self.log_filename, 
            methods=self.remote_methods(), cmd_event=self.cmd_event)
        self.data_proxy = DataPipe(data_pipe2)
        super().start()
        return self.target_proxy, self.data_proxy
    
    def is_enabled(self):
        return self.status.value > 0

    def disable(self):
        self.status.value = 0;

    def is_cmd_present(self):
        return self.cmd_event.is_set()

    def clear_cmd(self):
        self.cmd_event.clear()

    def loop_task(self):
        import time
        time.sleep(0.01)

    def run(self):
        self.log_str("RPCProcess.run")

        self.target_constr()

        try:
            while self.is_enabled():
                if not self.target.is_running():
                    self.log_str("Termination condition reached:")
                    self.disable()
                # else:
                #     self.log_str("run state: %s" % self.target.run_state)

                if self.is_cmd_present():
                    self.proc_rpc_command()
                    self.clear_cmd()

                self.loop_task()

            self.log_str("RPCProcess.run END OF WHILE LOOP")
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

    def proc_rpc_command(self):
        cmd = self.cmd_pipe.recv()
        self.log_str("Received command: %s" % str(cmd))

        if cmd is None:
            self.disable()
            return 

        self.log_str('remote command received: %s, %s, %s\n' % cmd)
        print('rpc target', self.target)
        try:
            fn_name, cmd_args, cmd_kwargs = cmd
            fn = getattr(self.target, fn_name)
            self.log_str("Function: " + str(fn))

            fn_output = fn(*cmd_args, **cmd_kwargs)
            self.cmd_pipe.send(fn_output)

            self.log_str("Target = %s" % str(self.target))
            self.log_str('self.target.__dict:' + str(self.target.__dict__))
            self.log_str('Done with command: %s, output=%s\n\n' % (fn_name, fn_output))
            return "completed"
        except Exception as e:
            err = io.StringIO()
            self.log_error(err, mode='a')

            self.cmd_pipe.send(e)
            return "error"

