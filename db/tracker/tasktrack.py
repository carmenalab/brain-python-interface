'''
Web browser GUI-launched tasks run in a separate process. This module provides 
mechanisms for interacting withe the task running in another process, e.g.,
calling functions to start/stop the task, enabling/disabling decoder adaptation, etc.
'''
import os
os.environ['DISPLAY'] = ':0'
import sys
import time
import xmlrpc.client
import multiprocessing as mp
import collections

from riglib import experiment, singleton
# from riglib.mp_proxy import FuncProxy, ObjProxy, RPCProcess2

from . import websocket

from config import config
from .json_param import Parameters

import io
import traceback

log_filename = os.path.join(config.log_path, "tasktrack_log")
def log_error(err, mode='a'):
    traceback.print_exc(None, err)
    with open(log_filename, mode) as fp:
        err.seek(0)
        fp.write(err.read())

def log_str(s, mode="a", newline=True):
    if newline and not s.endswith("\n"):
        s += "\n"
    with open(log_filename, mode) as fp:
        fp.write(s)


if sys.platform == "win32":
    use_websock = False
else:
    use_websock = True

class Track(singleton.Singleton):
    '''
    Tracker for task instantiation running in a separate process. This is a singleton.
    '''
    __instance = None 
    def __init__(self, use_websock=use_websock):
        super().__init__()
        # shared memory to store the status of the task in a char array
        self.status = mp.Array('c', 256)
        self.task_proxy = None
        # self.reset()
        self.proc = None
        # self.init_pipe()
        if use_websock:
            self.websock = websocket.Server(self.notify)
        else:
            self.websock = None

    # def init_pipe(self):
    #     self.tracker_end_of_pipe, self.task_end_of_pipe = mp.Pipe()

    def notify(self, msg):
        if msg['status'] == "error" or msg['State'] == "stopped":
            self.status.value = b""

    def runtask(self, base_class=experiment.Experiment, feats=[], **kwargs):
        '''
        Begin running of task
        '''
        log_str("Running new task: \n", mode="w")
        # self.init_pipe()

        if None in feats:
            raise ValueError("Features not found properly in database!")        

        # initialize task status
        # self.status.value = b"testing" if 'saveid' in kwargs else b"running"
        self.status.value = b"running" if 'saveid' in kwargs else b"testing" 

        # create a proxy for interacting with attributes/functions of the task.
        # The task runs in a separate process and we cannot directly access python 
        # attributes of objects in other processes
        # self.task_proxy = TaskObjProxy(self.tracker_end_of_pipe, log_filename)

        # Spawn the process
        # args = (self.tracker_end_of_pipe, self.task_end_of_pipe, self.websock)
        print("Track.runtask")

        if 'seq' in kwargs:
            kwargs['seq_params'] = kwargs['seq'].params
            kwargs['seq'] = kwargs['seq'].get()  ## retreive the database data on this end of the pipe

        use_websock = not (self.websock is None)        
        if use_websock:
            feats.insert(0, websocket.NotifyFeat)
            kwargs['params']['websock'] = self.websock
            kwargs['params']['tracker_status'] = self.status
        
        # kwargs['params']['tracker_end_of_pipe'] = tracker_end_of_pipe
        
        task_class = experiment.make(base_class, feats=feats)        

        # process parameters
        params = kwargs['params']
        if isinstance(params, str):
            params = Parameters(params)
        elif isinstance(params, dict):
            params = Parameters.from_dict(params)

        params.trait_norm(task_class.class_traits())

        params = params.params # dict
        kwargs.pop('params', None)

        
        # self.task_args = args
        # self.task_kwargs = kwargs

        log_str("Spawning process...")
        log_str(str(kwargs))

        self.proc = experiment.task_wrapper.TaskWrapper(
            log_filename=log_filename, params=params,
            target_class=task_class, websock=self.websock, status=self.status,
            **kwargs)
        self.task_proxy, _ = self.proc.start()

        # self.proc = mp.Process(target=remote_runtask, args=args, kwargs=kwargs)
        # self.proc.start()
        
    def __del__(self):
        '''
        Destructor for Track object. Not sure if this function ever gets called 
        since Track is a singleton created upon import of the db.tracker.ajax module...
        '''
        if self.websock is not None:
            self.websock.stop()

    # def pausetask(self):
    #     self.status.value = bytes(self.task_proxy.pause())

    def stoptask(self):
        '''
        Terminate the task gracefully by running riglib.experiment.Experiment.end_task
        '''
        assert self.status.value in [b"testing", b"running"]
        try:
            self.task_proxy.end_task()
        except Exception as e:
            traceback.print_exc()
            err = io.StringIO()
            traceback.print_exc(None, err)
            err.seek(0)
            return dict(status="error", msg=err.read())

        status = self.status.value.decode("utf-8")
        self.status.value = b""
        self.reset()
        return status

    def reset(self):
        self.task_proxy = None
        # self.task_kwargs = {}
        # self.task_args = ()        

    def get_status(self):
        return self.status.value.decode("utf-8")

    def update_alive(self):
        """ Check if the remote process is still alive, and if dead, reset the task_proxy object """
        if (not self.proc is None) and (not self.proc.is_alive()):
            print("process died in error, destroying proxy object")
            self.reset()

    def task_running(self):
        print(self.get_status())
        return self.get_status() in ["running", "testing"]

