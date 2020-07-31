import os
import sys
import time
import inspect
import traceback
import multiprocessing as mp
from multiprocessing import sharedctypes as shm
import io

import numpy as np


class FuncProxy(object):
    '''
    Interface for calling functions in remote processes.
    '''
    def __init__(self, name, pipe, event=None):
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
        self.pipe.send((self.name, args, kwargs))
        if not self.event is None:
            self.event.set()
        return self.pipe.recv()


class PipeWrapper(object):
    def __init__(self, pipe=None, log_filename='', **kwargs):
        self.pipe = pipe
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


class ObjProxy(PipeWrapper):
    def __getattr__(self, attr):
        self.log_str("remotely getting attribute: %s\n" % attr)

        self.pipe.send(("__getattribute__", [attr], {}))
        ret = self.pipe.recv()
        if isinstance(ret, Exception): 
            # Assume that the attribute can't be retreived b/c the name refers 
            # to a function
            ret = FuncProxy(attr, self.pipe)

        return ret

    def terminate(self):
        self.pipe.send(None)


class RPCPipe(PipeWrapper):
    def proc_rpc_command(self, cmd):
        print(cmd)
        self.log_str('remote command received: %s, %s, %s\n' % cmd)
        try:
            fn_name, cmd_args, cmd_kwargs = cmd
            fn = getattr(self, fn_name)

            fn_output = fn(*cmd_args, **cmd_kwargs)
            self.pipe.send(fn_output)

            self.log_str('Done with command: %s\n\n' % fn_name)            
            return "completed"
        except Exception as e:
            err = io.StringIO()
            self.log_error(err, mode='a')

            self.pipe.send(e)
            return "error"


class RPCProcess(mp.Process):
    """mp.Process which implements remote procedure call (RPC) through a mp.Pipe object"""
    proxy = ObjProxy
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pipe = None
        self.log_filename = kwargs.pop('log_filename', '')
        self.run_the_loop = False

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

    def start(self):
        self.pipe, pipe_end2 = mp.Pipe()
        super().start()
        return self.proxy(pipe=pipe_end2)
        # return ObjProxy(pipe=pipe_end2)

    def check_run_condition(self):
        return True

    def run(self):
        self.run_the_loop = True
        try:
            self.log_str("RPCProcess.run")
            while self.run_the_loop:
                if not self.check_run_condition():
                    self.log_str("Child termination condition reached: %s" % self.task.state)
                    break

                if self.pipe.poll(1):
                    cmd = self.pipe.recv()
                    self.log_str("Received command: %s" % str(cmd))

                    if cmd is None:
                        self.run_the_loop = False
                        continue

                    proc_status = self.proc_rpc_command(cmd)
                    if proc_status == "completed":
                        cmd = self.pipe.recv()
                    elif proc_status == "error":
                        if self.pipe.poll(60.):
                            cmd = self.pipe.recv()
                        else:
                            self.run_the_loop = False
            self.log_str("RPCProcess.run END OF WHILE LOOP")
            return 0, ''
        except KeyboardInterrupt:
            return 1, 'KeyboardInterrupt'
        except:
            import traceback
            traceback.print_exc()
            err = io.StringIO()
            self.log_error(err, mode='a')
            err.seek(0)
            return 1, err.read()

    def proc_rpc_command(self, cmd):
        print("command", cmd)
        self.log_str("proc_rpc_command")
        if cmd is None:
            self.run_the_loop = False
            return 

        self.log_str('remote command received: %s, %s, %s\n' % cmd)
        try:
            fn_name, cmd_args, cmd_kwargs = cmd
            fn = getattr(self, fn_name)

            fn_output = fn(*cmd_args, **cmd_kwargs)
            self.pipe.send(fn_output)

            self.log_str('Done with command: %s\n\n' % fn_name)            
            return "completed"
        except Exception as e:
            err = io.StringIO()
            self.log_error(err, mode='a')

            self.pipe.send(e)
            return "error"
