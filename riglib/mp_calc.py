import multiprocessing as mp
import time
import numpy as np
import queue

class MPCompute(mp.Process):
    """
    Generic class for running computations that occur infrequently
    but take longer than a single BMI loop iteration
    """
    def __init__(self, work_queue, result_queue, fn):
        '''
        Constructor for MPCompute

        Parameters
        ----------
        work_queue : mp.Queue
            Jobs start when an entry is found in work_queue
        result_queue : mp.Queue
            Results of job are placed back onto result_queue

        Returns
        -------
        MPCompute instance
        '''
        # run base constructor
        super(MPCompute, self).__init__()

        self.work_queue = work_queue
        self.result_queue = result_queue
        self.done = mp.Event()
        self.fn = fn

    def _check_for_job(self):
        '''
        Non-blocking check to see if data is present in the input queue
        '''
        try:
            job = self.work_queue.get_nowait()
        except:
            job = None
        return job
        
    def run(self):
        '''
        The main loop. Starts automatically when the process is spawned. See mp.Process.run for additional docs.
        Every 0.5 seconds, check for new computation to carry out

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        while not self.done.is_set():
            job = self._check_for_job()

            # unpack the data
            if not (job is None):
                new_params = self.calc(*job[0], **job[1])
                self.result_queue.put(new_params)

            # Pause to lower the process's effective priority
            time.sleep(0.5)

    def calc(self, *args, **kwargs):
        '''
        Run the actual calculation function
        '''
        return self.fn(*args, **kwargs)

    def stop(self):
        '''
        Set the flag to stop the 'while' loop in the 'run' method gracefully
        '''
        self.done.set()


class FuncProxy(object):
    '''
    Wrapper for MPCompute computations running in another process
    '''
    def __init__(self, fn, multiproc=False, waiting_resp=None, init_resp=None, verbose=False):
        self.verbose = verbose
        self.multiproc = multiproc
        if self.multiproc:
            # create the queues
            self.work_queue = mp.Queue()
            self.result_queue = mp.Queue()

            # Instantiate the process
            self.calculator = MPCompute(self.work_queue, self.result_queue, fn)

            # spawn the process
            self.calculator.start()
        else:
            self.fn = fn

        assert waiting_resp in [None, 'prev'], "Unrecognized waiting_resp"
        self.waiting_resp = waiting_resp

        self.prev_result = (init_resp, 0)
        self.prev_input = None
        self.waiting = False

    def reset(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.prev_input = None

    def _stuff(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        try:
            output_data = self.result_queue.get_nowait()
            self.prev_result = output_data
            self.waiting = False
            return output_data, True
        except queue.Empty:
            if self.waiting_resp == None:
                return None
            elif self.waiting_resp == 'prev':
                return self.prev_result, False
        except:
            import traceback
            traceback.print_exc()

    def input_same(self, stuff):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        args, kwargs = stuff
        if self.prev_input == None:
            return False

        args_same = True
        for a1, a2 in zip(args, self.prev_input[0]):
            try: 
                args_same = args_same and np.all(a1 == a2)
            except ValueError:
                args_same = args_same and np.array_equal(a1, a2)

        kwargs_same = list(kwargs.keys()) == list(self.prev_input[1].keys())

        for key1, key2 in zip(list(kwargs.keys()), list(self.prev_input[1].keys())):
            k1 = kwargs[key1]
            k2 = kwargs[key2]
            if key1 == 'q_start': 
                continue
            try:
                kwargs_same = kwargs_same and np.all(k1 == k2)
            except:
                kwargs_same = kwargs_same and np.array_equal(k1, k2)

        return args_same and kwargs_same


    def __call__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        input_data = (args, kwargs)
        input_same_as_last = self.input_same(input_data) #input_data == self.prev_input        
        if self.multiproc:
            if input_same_as_last and not self.waiting:
                return self.prev_result, False

            elif input_same_as_last and self.waiting:
                # Return the new result if it's available, otherwise the previous result
                return self._stuff()

            elif not input_same_as_last:
                if self.verbose: print("queuing job")
                self.work_queue.put(input_data)    
                self.prev_input = input_data
                self.waiting = True
                return self._stuff()
        else:
            if input_same_as_last:
                return self.prev_result, False
            else:
                self.prev_input = input_data
                self.prev_result = self.fn(*args, **kwargs)
                return self.prev_result, True

    def __del__(self):
        '''
        Stop the child process if one was spawned
        '''
        if self.multiproc:
            self.calculator.stop()

class MultiprocShellCommand(mp.Process):
    '''
    Execute a blocking shell command in a separate process
    '''
    def __init__(self, cmd, *args, **kwargs):
        self.cmd = cmd
        self.done = mp.Event()
        super(MultiprocShellCommand, self).__init__(*args, **kwargs)

    def run(self):
        '''
        Docstring
        '''
        import os
        os.popen(self.cmd)
        self.done.set()

    def is_done(self):
        return self.done.is_set()