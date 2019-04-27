'''
Web browser GUI-launched tasks run in a separate process. This module provides 
mechanisms for interacting withe the task running in another process, e.g.,
calling functions to start/stop the task, enabling/disabling decoder adaptation, etc.
'''

import os
import sys
import time
import xmlrpc.client
import multiprocessing as mp
import collections

from riglib import experiment
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

def log_str(s, mode="a"):
    with open(log_filename, mode) as fp:
        fp.write(s)


class Track(object):
    '''
    Tracker for task instantiation running in a separate process. This is a singleton.
    '''
    def __init__(self, use_websock=True):
        # shared memory to store the status of the task in a char array
        self.status = mp.Array('c', 256)
        self.task_proxy = None
        self.proc = None
        self.tracker_end_of_pipe, self.task_end_of_pipe = mp.Pipe()
        if use_websock:
            self.websock = websocket.Server(self.notify)
        else:
            self.websock = None

    def notify(self, msg):
        if msg['status'] == "error" or msg['State'] == "stopped":
            self.status.value = b""

    def runtask(self, **kwargs):
        '''
        Begin running of task
        '''
        with open(log_filename, 'w') as f:
            f.write("Running new task: \n")

        # initialize task status
        # self.status.value = b"testing" if 'saveid' in kwargs else b"running"
        self.status.value = b"running" if 'saveid' in kwargs else b"testing" 

        # create a proxy for interacting with attributes/functions of the task.
        # The task runs in a separate process and we cannot directly access python 
        # attributes of objects in other processes
        self.task_proxy = TaskObjProxy(self.tracker_end_of_pipe)

        # Spawn the process
        args = (self.tracker_end_of_pipe, self.task_end_of_pipe, self.websock)
        print("Track.runtask")
        print(kwargs)

        if 'seq' in kwargs:
            kwargs['seq_params'] = kwargs['seq'].params
            kwargs['seq'] = kwargs['seq'].get()  ## retreive the database data on this end of the pipe
            print(kwargs['seq'])

        print(args)
        print(kwargs)
        kwargs.pop("params")
        self.proc = mp.Process(target=remote_runtask, args=args)
        self.proc.start()
        
    def __del__(self):
        '''
        Destructor for Track object. Not sure if this function ever gets called 
        since Track is a singleton created upon import of the db.tracker.ajax module...
        '''
        if not self.websock is None:
            self.websock.stop()

    def pausetask(self):
        self.status.value = bytes(self.task_proxy.pause())

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
        self.task_proxy = None
        return status

    def update_alive(self):
        if (not self.proc is None) and (not self.proc.is_alive()):
            print("process died in error, destroying proxy object")
            self.task_proxy = None


def remote_runtask(tracker_end_of_pipe, task_end_of_pipe, websock, **kwargs):
    '''
    Target function to execute in the spawned process to start the task
    '''
    print("*************************** STARTING TASK *****************************")
    
    use_websock = not (websock is None)

    # Rerout prints to stdout to the websocket
    if use_websock:
        sys.stdout = websock

    # os.nice sets the 'niceness' of the task, i.e. how willing the process is
    # to share resources with other OS processes. Zero is neutral
    os.nice(0)

    status = "running" if 'saveid' in kwargs else "testing"

    # Force all tasks to use the Notify feature defined above. 
    if use_websock:
        kwargs['params']['websock'] = websock
        kwargs['feats'].insert(0, websocket.NotifyFeat)
    kwargs['params']['tracker_status'] = status
    kwargs['params']['tracker_end_of_pipe'] = tracker_end_of_pipe        

    try:
        # Instantiate the task
        task_wrapper = TaskWrapper(**kwargs)
        cmd = task_end_of_pipe.recv()

        # Rerout prints to stdout to the websocket
        if use_websock: sys.stdout = websock

        while (cmd is not None) and (task_wrapper.task.state is not None):
            log_str('remote command received: %s, %s, %s\n' % cmd)
            try:
                fn_name = cmd[0]
                cmd_args = cmd[1]
                cmd_kwargs = cmd[2]

                # look up the function by name
                fn = getattr(task_wrapper, fn_name)

                # run the function and save the return value as a single object
                # if an exception is thrown, the code will jump to the last 'except' case
                ret = fn(*cmd_args, **cmd_kwargs)
                log_str("return value: %s\n" % str(ret))

                # send the return value back to the remote process 
                task_end_of_pipe.send(ret)

                # hang and wait for the next command to come in
                log_str("task state = %s, stop status=%s, waiting for next command...\n" % (task_wrapper.task.state, str(task_wrapper.task.stop)))
                cmd = task_end_of_pipe.recv()
            except KeyboardInterrupt:
                # Handle the KeyboardInterrupt separately. How the hell would
                # a keyboard interrupt even get here?
                cmd = None
            except Exception as e:
                err = io.StringIO()
                log_error(err, mode='a')

                task_end_of_pipe.send(e)
                if task_end_of_pipe.poll(60.):
                    cmd = task_end_of_pipe.recv()
                else:
                    cmd = None
            log_str('Done with command: %s\n\n' % fn_name)            
    except:
        task_wrapper = None
        err = io.StringIO()
        log_error(err, mode='a')
        err.seek(0)
        if use_websock:
            websock.send(dict(status="error", msg=err.read()))
        with open(log_filename, 'a') as f:
            err.seek(0)
            f.write(err.read())
        err.seek(0)
        print(err.read())

    log_str('End of task while loop\n')

    # Redirect printing from the websocket back to the shell
    if use_websock:
        websock.write("Running task cleanup functions....\n")
    sys.stdout = sys.__stdout__
    print("Running task cleanup functions....\n")

    # Initiate task cleanup
    if task_wrapper is None:
        print("\nERROR: Task was never initialized, cannot run cleanup function!")
        print("see %s for error messages" % log_filename)
        print(open(log_filename, 'rb').read())
        print()

        if 'saveid' in kwargs:
            from . import dbq
            dbq.hide_task_entry(kwargs['saveid'])
            print('hiding task entry!')
        
        cleanup_successful = False
    else:
        cleanup_successful = task_wrapper.cleanup()


    # inform the user in the browser that the task is done!
    if cleanup_successful:
        if use_websock: websock.write("\n\n...done!\n")
    else:
        if use_websock: websock.write("\n\nError! Check for errors in the terminal!\n")

    print("*************************** EXITING TASK *****************************")


class TaskWrapper(object):
    '''
    Wrapper for Experiment classes launched from the web interface
    '''
    def __init__(self, subj, task_rec, feats, params, seq=None, seq_params=None, saveid=None):
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
        seq : models.Sequence instance, or tuple
            Database record of Sequence parameters/static target sequence
            If passed in as a tuple, then it's the result of calling 'seq.get' on the models.Sequence instance
        seq_params: params from seq (see above)

        saveid : int, optional
            ID number of db.tracker.models.TaskEntry associated with this task
            if None specified, then the data saved will not be linked to the
            database entry and will be lost after the program exits
        '''
        self.saveid = saveid
        self.taskname = task_rec.name
        self.subj = subj
        if isinstance(params, Parameters):
            self.params = params
        elif isinstance(params, (string, str)):
            self.params = Parameters(params)
        
        base_class = task_rec.get()

        if None in feats:
            raise Exception("Features not found properly in database!")
        else:
            Task = experiment.make(base_class, feats=feats)

        # Run commands which must be executed before the experiment class can be instantiated (e.g., starting neural recording)
        Task.pre_init(saveid=saveid)

        self.params.trait_norm(Task.class_traits())
        if issubclass(Task, experiment.Sequence):
            from . import models
            # retreive the sequence data from the db, or from the input argument if the input arg was a tuple
            if isinstance(seq, tuple):
                gen_constructor, gen_params = seq
            elif isinstance(seq, models.Sequence):
                gen_constructor, gen_params = seq.get()
                # Typically, 'gen_constructor' is the experiment.generate.runseq function (not an element of namelist.generators)
            else:
                raise ValueError("Unrecognized type for seq")

            gen = gen_constructor(Task, **gen_params)
            self.params.params['seq_params'] = seq_params
            
            # 'gen' is now a true python generator usable by experiment.Sequence
            self.task = Task(gen, **self.params.params)

            with open(log_filename, 'a') as f:
                f.write("instantiating task with a generator\n")

        else:
            self.task = Task(**self.params.params)
        self.task.start()

    def report(self):
        return experiment.report(self.task)
    
    def pause(self):
        self.task.pause = not self.task.pause
        return "pause" if self.task.pause else "running"
    
    def end_task(self):
        return self.task.end_task()

    def enable_clda(self):
        self.task.enable_clda()

    def disable_clda(self):
        self.task.disable_clda()
    
    def get_state(self):
        return self.task.state

    def __getattr__(self, attr):
        # This function is only defined because __getattr__ is not defined 
        # for children of 'object' by default, but the TaskObjProxy always calles '__getattr__'
        # when trying to remotely retreive an attribute. Might be avoidable if TaskObjProxy were
        # to use '__getattribute__' instead
        return getattr(self, attr)

    def set_task_attr(self, attr, value):
        setattr(self.task, attr, value)

    def cleanup(self):
        self.task.join()
        print("Calling saveout/task cleanup code")
        
        if self.saveid is not None:
            # get object representing function calls to the remote database
            # returns the result of tracker.dbq.rpc_handler
            # database = xmlrpc.client.ServerProxy("http://localhost:8000/RPC2/", allow_none=True)
            from . import dbq

            cleanup_successful = self.task.cleanup(dbq, self.saveid, subject=self.subj)
            
            # if not self.task._task_init_complete:
            #     from tracker import dbq
            #     dbq.hide_task_entry(self.saveid)
            #     print 'hiding task entry!'
            # else:
            #     print 'not hiding task entry!'
        else:
            cleanup_successful = True

        self.task.terminate()
        return cleanup_successful


class TaskObjProxy(object):
    def __init__(self, tracker_end_of_pipe):
        self.tracker_end_of_pipe = tracker_end_of_pipe

    def __getattr__(self, attr):
        with open(log_filename, 'a') as f:
            f.write("remotely getting attribute: %s\n" % attr)

        self.tracker_end_of_pipe.send(("__getattr__", [attr], {}))
        ret = self.tracker_end_of_pipe.recv()
        if isinstance(ret, Exception): 
            # Assume that the attribute can't be retreived b/c the name refers 
            # to a function
            ret = TaskFuncProxy(attr, self.tracker_end_of_pipe)

        return ret

    def end_task(self):
        end_task_fn = TaskFuncProxy("end_task", self.tracker_end_of_pipe)
        end_task_fn()
        self.tracker_end_of_pipe.send(None)

    def remote_set_attr(self, attr, value):
        log_str('trying to remotely set attribute %s to %s\n' % (attr, value))
        ret = TaskFuncProxy('set_task_attr', self.tracker_end_of_pipe)
        ret(attr, value)


class TaskFuncProxy(object):
    def __init__(self, func, pipe):
        self.pipe = pipe
        self.cmd = func

    def __call__(self, *args, **kwargs):
        self.pipe.send((self.cmd, args, kwargs))
        return self.pipe.recv()