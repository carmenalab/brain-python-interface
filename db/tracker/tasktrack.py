'''
Web browser GUI-launched tasks run in a separate process. This module provides
mechanisms for interacting withe the task running in another process, e.g.,
calling functions to start/stop the task, enabling/disabling decoder adaptation, etc.
'''
import os
import sys
import io
import traceback
from datetime import datetime
import multiprocessing as mp

from . import websocket
from riglib import experiment, singleton
from .json_param import Parameters

log_path = os.path.join(os.path.dirname(__file__), '../../log')
log_filename = os.path.join(log_path, "tasktrack_log")
# TODO make folder if it doesn't exist

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
        self.proc = None
        self.use_websock = use_websock
        self.websock = None

    def start_websock(self):
        if self.websock is None and self.use_websock:
            print("Starting websocket...")
            self.websock = websocket.Server(self.notify)

    def notify(self, msg):
        if msg['status'] == "error" or msg['State'] == "stopped":
            self.status.value = b""

    def runtask(self, base_class=experiment.Experiment, feats=[], cli=False, **kwargs):
        '''
        Begin running of task
        '''
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        log_str("\n{}\n-------------------\nRunning new task: {}\n".format(now, base_class))

        if not cli:
            self.start_websock()

        if None in feats:
            raise ValueError("Features not found properly in database!")

        # initialize task status
        self.status.value = b"running" if 'saveid' in kwargs else b"testing"

        if 'seq' in kwargs:
            kwargs['seq_params'] = kwargs['seq'].params
            kwargs['seq'] = kwargs['seq'].get()  ## retreive the database data on this end of the pipe

        use_websock = not (self.websock is None)
        if use_websock:
            feats.insert(0, websocket.NotifyFeat)
            kwargs['params']['websock'] = self.websock
            kwargs['params']['tracker_status'] = self.status

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

        log_str("Spawning process...")
        log_str(str(kwargs))

        # Spawn the process
        self.proc = experiment.task_wrapper.TaskWrapper(
            log_filename=log_filename, params=params,
            target_class=task_class, websock=self.websock, status=self.status,
            **kwargs)

        # create a proxy for interacting with attributes/functions of the task.
        # The task runs in a separate process and we cannot directly access python
        # attributes of objects in other processes
        self.task_proxy, _ = self.proc.start()

    def __del__(self):
        '''
        Destructor for Track object. Not sure if this function ever gets called
        since Track is a singleton created upon import of the db.tracker.ajax module...
        '''
        if self.websock is not None:
            self.websock.stop()

    def stoptask(self):
        '''
        Terminate the task gracefully by running riglib.experiment.Experiment.end_task
        '''
        self.update_alive()            
        if self.status.value == b"":
            print("Task already stopped!")
            return
        elif self.status.value not in [b"testing", b"running"]:
            raise ValueError("Unknown task status")

        assert self.status.value in [b"testing", b"running"]
        print("Tasktrack is stopping the task...")
        try:
            if self.task_proxy is not None:
                self.task_proxy.end_task()
        except Exception as e:
            print(e)
            traceback.print_exc()
            err = io.StringIO()
            traceback.print_exc(None, err)
            err.seek(0)
            return dict(status="error", msg=err.read())

        status = self.status.value.decode("utf-8")
        self.reset()

        '''
        WIP cloud upload code
        # allow time for files, etc. to be saved
        time.sleep(3)

        if self.proc.saveid is not None:
            from db.tracker import models
            te = models.TaskEntry.objects.get(id=self.proc.saveid)

            # Wrap up HDF file saving
            models.DataFile.objects.get(entry__id=te.id, system__name="hdf")
            metadata = dict(
                task_name = te.task.name,
                block_number = te.id,
            )

            # upload metadata to server, if appropriate
            print("Attempting to save to cloud...")
            te.upload_to_cloud()
        '''

        return status

    def reset(self):
        self.task_proxy = None
        self.status.value = b""

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