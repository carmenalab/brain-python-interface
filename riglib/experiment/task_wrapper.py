import multiprocessing as mp
import sys
import numpy as np
import os
import time
import xmlrpc.client

from . import Experiment, Sequence
from ..mp_proxy import FuncProxy, ObjProxy, RPCProcess


class TaskWrapper(RPCProcess):
    '''
    Wrapper for Experiment classes launched from the web interface
    '''
    proxy = ObjProxy

    def __init__(self, subj, subject_name, params, target_class=Experiment,
        websock=None, status='',
        seq=None, seq_params=None, saveid=None, **kwargs):
        '''
        Parameters
        ----------
        subj : tracker.models.Subject instance
            Database record for subject performing the task
        base_class : a child class of riglib.experiment.Experiment
            The base class for the task, without the feature mixins
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
        print("TaskWrapper.__init__", kwargs)
        super().__init__(**kwargs)
        self.log_str("TaskWrapper constructor")
        self.subj = subj
        self.subject_name = subject_name
        self.target_class = target_class
        self.params = params
        self.seq = seq
        self.seq_params = seq_params
        self.saveid = saveid
        self.websock = websock
        self.task_status = status

    def target_constr(self):
        self.log_str("TaskWrapper target construction")
        seq = self.seq
        saveid = self.saveid

        Task = self.target_class
        self.params['subject_name'] = self.subject_name

        # Run commands which must be executed before the experiment class can be instantiated (e.g., starting neural recording)
        self.target_class.pre_init(saveid=saveid, **self.params)

        self.log_str("Constructing task target")
        self.params['saveid'] = saveid

        if issubclass(Task, Sequence):
            if isinstance(seq, np.ndarray):
                self.log_str("'seq' input is an array")
                task = Task(seq, **self.params)
                self.log_str("instantiating task with a sequence\n")
            else:
                # retreive the sequence data from the db, or from the input argument if the input arg was a tuple
                if isinstance(seq, tuple):
                    gen_constructor, gen_params = seq
                elif hasattr(seq, 'get'):
                    gen_constructor, gen_params = seq.get()
                    # Typically, 'gen_constructor' is the experiment.generate.runseq function (not an element of namelist.generators)
                else:
                    raise ValueError("Unrecognized type for seq")

                gen = gen_constructor(Task, **gen_params)
                self.params['seq_params'] = self.seq_params

                # 'gen' is now a true python generator usable by experiment.Sequence
                task = Task(gen, **self.params)
                self.log_str("instantiating task with a generator\n")
        else:
            task = Task(**self.params)


        self.log_str("TaskWrapper.task initialized")

        # Rerout prints to stdout to the websocket
        if not (self.websock is None):
            sys.stdout = self.websock

        # os.nice sets the 'niceness' of the task, i.e. how willing the process is
        # to share resources with other OS processes. Zero is neutral
        if not sys.platform == "win32":
            os.nice(0)

        self.task_status = "running" if saveid is not None else "testing"

        self.log_str("Initial task state: %s" % task.state)

        print("*************************** STARTING TASK *****************************")

        self.target = task
        self.target.start()

    def target_destr(self, ret_status, msg):
        self.log_str('End of task while loop (target_destr fn)')
        use_websock = not (self.websock is None)
        if ret_status != 0:
            if use_websock:
                self.websock.send(dict(status="error", msg=msg))
            self.log_str(msg)

        # Redirect printing from the websocket back to the shell
        if use_websock:
            self.websock.write("Running task cleanup functions....\n")
        sys.stdout = sys.__stdout__
        print("Running task cleanup functions....\n")

        self.log_str("Starting cleanup...")
        cleanup_successful = self.cleanup()

        # inform the user in the browser that the task is done!         TODO: Get errors to show up as errors in the web UI
        if cleanup_successful == True or cleanup_successful is None:
            if use_websock: self.websock.write("\n\n...done!\n")
        else:
            if use_websock: self.websock.write("\n\nError! Check for errors in the terminal!\n")

        self.log_str("...done")
        print("*************************** EXITING TASK *****************************")

    def check_run_condition(self):
        return self.target.state is not None

    def report(self):
        return self.target_proxy.online_report()

    def cleanup(self):
        self.target.join()
        print("Calling saveout/task cleanup code")

        if self.saveid is not None and self.subj is not None:
            try:
                # get object representing function calls to the remote database
                # returns the result of db.tracker.dbq.rpc_handler
                database = xmlrpc.client.ServerProxy("http://localhost:8000/RPC2/", allow_none=True)
                # from tracker import dbq as database
                cleanup_successful = self.target.cleanup(database, self.saveid, subject=self.subj)
                database.cleanup(self.saveid)
            except Exception as e:
                self.log_str("Error in cleanup:")
                self.log_error(e)
                cleanup_successful = False
        else:
            cleanup_successful = True

        self.target.terminate()
        return cleanup_successful

    def stop(self):
        self.target_proxy.end_task()
        super().stop()

    def get_state(self):
        return self.target.state