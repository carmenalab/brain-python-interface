import multiprocessing as mp
import time
from itertools import izip
import numpy as np
import Queue

class MPCompute(mp.Process):
    """
    Generic class for running computations that occur infrequently
    but take longer than a single BMI loop iteration
    """
    def __init__(self, work_queue, result_queue, fn):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------


        __init__

        work_queue, result_queue are mp.Queues
        Jobs start when an entry is found in work_queue
        Results of job are placed back onto result_queue
        '''
        # run base constructor
        super(MPCompute, self).__init__()

        self.work_queue = work_queue
        self.result_queue = result_queue
        self.done = mp.Event()
        self.fn = fn

    def _check_for_job(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        try:
            job = self.work_queue.get_nowait()
        except:
            job = None
        return job
        
    def run(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        """
        The main loop. Starts automatically when 
        """
        while not self.done.is_set():
            job = self._check_for_job()

            # unpack the data
            if not job == None:
                new_params = self.calc(*job[0], **job[1])
                self.result_queue.put(new_params)

            # Pause to lower the process's effective priority
            time.sleep(0.5)

    def calc(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return self.fn(*args, **kwargs)

    def stop(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.done.set()

class FuncProxy(object):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
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
        except Queue.Empty:
            if self.waiting_resp == None:
                return None
            elif self.waiting_resp == 'prev':
                return self.prev_result, False
        except:
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
        for a1, a2 in izip(args, self.prev_input[0]):
            try: 
                args_same = args_same and np.all(a1 == a2)
            except ValueError:
                args_same = args_same and np.array_equal(a1, a2)

        kwargs_same = kwargs.keys() == self.prev_input[1].keys()

        for key1, key2 in izip(kwargs.keys(), self.prev_input[1].keys()):
            k1 = kwargs[key1]
            k2 = kwargs[key2]
            if key1 == 'q_start': 
                continue
        # for k1, k2 in izip(kwargs.values(), self.prev_input[1].values()):
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
                if self.verbose: print "queuing job"
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
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        # Stop the child process if one was spawned
        if self.multiproc:
            self.calculator.stop()
