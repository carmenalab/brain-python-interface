'''
Web browser GUI-launched tasks run in a separate process. This module provides 
mechanisms for interacting withe the task running in another process, e.g.,
calling functions to start/stop the task, enabling/disabling decoder adaptation, etc.
'''

import os
import sys
import time
import xmlrpclib
import multiprocessing as mp
import collections

from riglib import experiment
import websocket

from config import config
from json_param import Parameters

import cStringIO
import traceback

log_filename = os.path.expandvars('$BMI3D/log/tasktrack_log')
def log_error(err, mode='a'):
    traceback.print_exc(None, err)
    with open(log_filename, mode) as fp:
        err.seek(0)
        fp.write(err.read())


class Track(object):
    '''
    Tracker for task instantiation running in a separate process
    '''
    def __init__(self):
        # shared memory to store the status of the task in a char array
        self.status = mp.Array('c', 256)
        self.task_proxy = None
        self.proc = None
        self.websock = websocket.Server(self.notify)
        self.tracker_end_of_pipe, self.task_end_of_pipe = mp.Pipe()

    def notify(self, msg):
        if msg['status'] == "error" or msg['State'] == "stopped":
            self.status.value = ""

    def runtask(self, **kwargs):
        '''
        Begin running of task
        '''
        with open(log_filename, 'w') as f:
            f.write("Running new task: \n")

        # initialize task status
        self.status.value = "testing" if 'saveid' in kwargs else "running"

        # create a proxy for interacting with attributes/functions of the task.
        # The task runs in a separate process and we cannot directly access python 
        # attributes of objects in other processes
        self.task_proxy = TaskObjProxy(self.tracker_end_of_pipe)

        # Spawn the process
        args = (self.tracker_end_of_pipe, self.task_end_of_pipe, self.websock)
        self.proc = mp.Process(target=remote_runtask, args=args, kwargs=kwargs)
        self.proc.start()
        
    def __del__(self):
        '''
        Destructor for Track object. Not sure if this function ever gets called 
        since Track is a singleton created upon import of the db.tracker.ajax module...
        '''
        self.websock.stop()

    def pausetask(self):
        self.status.value = self.task_proxy.pause()

    def stoptask(self):
        '''
        Terminate the task gracefully by running riglib.experiment.Experiment.end_task
        '''
        assert self.status.value in "testing,running"
        try:
            self.task_proxy.end_task()
        except Exception as e:
            err = cStringIO.StringIO()
            traceback.print_exc(None, err)
            err.seek(0)
            return dict(status="error", msg=err.read())

        status = self.status.value
        self.status.value = ""
        self.task_proxy = None
        return status


def remote_runtask(tracker_end_of_pipe, task_end_of_pipe, websock, **kwargs):
    '''
    Target function to execute in the spawned process to start the task
    '''
    print "*************************** STARTING TASK *****************************"
    
    # Rerout prints to stdout to the websocket
    sys.stdout = websock    

    # os.nice sets the 'niceness' of the task, i.e. how willing the process is
    # to share resources with other OS processes. Zero is neutral
    os.nice(0)

    status = "running" if 'saveid' in kwargs else "testing"

    # Force all tasks to use the Notify feature defined above. 
    kwargs['params']['websock'] = websock
    kwargs['params']['tracker_status'] = status
    kwargs['params']['tracker_end_of_pipe'] = tracker_end_of_pipe
    kwargs['feats'].insert(0, websocket.NotifyFeat)

    try:
        # Instantiate the task
        task_wrapper = TaskWrapper(**kwargs)
        cmd = task_end_of_pipe.recv()

        # Rerout prints to stdout to the websocket
        sys.stdout = websock

        while (cmd is not None) and (task_wrapper.task.state is not None):
            with open(log_filename, 'a') as f:
                f.write('remote command received: %s, %s, %s\n' % cmd)
            try:
                fn_name = cmd[0]
                cmd_args = cmd[1]
                cmd_kwargs = cmd[2]

                # look up the function by name
                fn = getattr(task_wrapper, fn_name)

                # run the function and save the return value as a single object
                # if an exception is thrown, the code will jump to the last 'except' case
                ret = fn(*cmd_args, **cmd_kwargs)

                # send the return value back to the remote process 
                task_end_of_pipe.send(ret)

                # hang and wait for the next command to come in
                cmd = task_end_of_pipe.recv()
            except KeyboardInterrupt:
                # Handle the KeyboardInterrupt separately. How the hell would
                # a keyboard interrupt even get here?
                cmd = None
            except Exception as e:
                task_end_of_pipe.send(e)
                if task_end_of_pipe.poll(60.):
                    cmd = task_end_of_pipe.recv()
                else:
                    cmd = None
    except:
        task_wrapper = None
        err = cStringIO.StringIO()
        log_error(err, mode='a')
        err.seek(0)
        websock.send(dict(status="error", msg=err.read()))
        err.seek(0)
        print err.read()

    # Redirect printing from the websocket back to the shell
    sys.stdout = sys.__stdout__

    # Initiate task cleanup
    if task_wrapper is None:
        print "\nERROR: Task was never initialized, cannot run cleanup function!"
        print "see %s for error messages" % log_filename
        print open(log_filename, 'rb').read()
        print
        
    else:
        task_wrapper.cleanup()

    print "*************************** EXITING TASK *****************************"


class TaskWrapper(object):
    '''
    Wrapper for Experiment classes launched from the web interface
    '''
    def __init__(self, subj, task_rec, feats, params, seq=None, saveid=None):
        '''
        Parameters
        ----------
        subj : tracker.models.Subject instance
            Database record for subject performing the task
        task_rec : tracker.models.Task instance
            Database record for base task being run (without features)
        feats : list 
            List of features to enable for the task
        params : json_param.Parameters, or string representation of JSON object
            user input on configurable task parameters
        seq : models.Sequence instance
            Database record of Sequence parameters/static target sequence
        saveid : int, optional
            ID number of db.tracker.models.TaskEntry associated with this task
            if None specified, then the data saved will not be linked to the
            database entry and will be lost after the program exits
        '''
        self.saveid = saveid
        self.taskname = task.name
        self.subj = subj
        if isinstance(params, Parameters):
            self.params = params
        elif isinstance(params, (string, unicode)):
            self.params = Parameters(params)
        
        base_class = task_rec.get()
        Task = experiment.make(base_class, feats=feats)

        # Run commands which must be executed before the experiment class can be instantiated (e.g., starting neural recording)
        Task.pre_init(saveid=saveid)

        self.params.trait_norm(Task.class_traits())
        if issubclass(Task, experiment.Sequence):
            gen_constructor, gen_params = seq.get()

            # Typically, 'gen_constructor' is the experiment.generate.runseq function (not an element of namelist.generators)
            gen = gen_constructor(Task, **gen_params)

            # 'gen' is now a true python generator usable by experiment.Sequence
            self.task = Task(gen, **self.params.params)
        else:
            self.task = Task(**self.params.params)
        
        self.task.start()

    def report(self):
        return experiment.report(self.task)
    
    def pause(self):
        self.task.pause = not self.task.pause
        return "pause" if self.task.pause else "running"
    
    def end_task(self):
        self.task.end_task()

    def enable_clda(self):
        self.task.enable_clda()

    def disable_clda(self):
        self.task.disable_clda()
    
    def get_state(self):
        return self.task.state

    def __getattr__(self, attr):
        return getattr(self, attr)

    def set_task_attr(self, attr, value):
        setattr(self.task, attr, value)

    def cleanup(self):
        self.task.join()
        print "Calling saveout/task cleanup code"
        
        if self.saveid is not None:
            # get object representing function calls to the remote database
            # returns the result of tracker.dbq.rpc_handler
            database = xmlrpclib.ServerProxy("http://localhost:8000/RPC2/", allow_none=True)

            self.task.cleanup(database, self.saveid, subject=self.subj)

        self.task.terminate()


class TaskObjProxy(object):
    def __init__(self, cmds):
        self.cmds = cmds

    def __getattr__(self, attr):
        with open(log_filename, 'a') as f:
            f.write("remotely getting attribute\n")

        self.cmds.send(("__getattr__", [attr], {}))
        ret = self.cmds.recv()
        if isinstance(ret, Exception): 
            # Assume that the attribute can't be retreived b/c the name refers 
            # to a function
            ret = TaskFuncProxy(attr, self.cmds)

        return ret

    def remote_set_attr(self, attr, value):
        with open(log_filename, 'a') as f:
            f.write('trying to remotely set attribute %s to %s\n' % (attr, value))
        ret = TaskFuncProxy('set_task_attr', self.cmds)
        ret(attr, value)


class TaskFuncProxy(object):
    def __init__(self, func, pipe):
        self.pipe = pipe
        self.cmd = func

    def __call__(self, *args, **kwargs):
        self.pipe.send((self.cmd, args, kwargs))
        return self.pipe.recv()
