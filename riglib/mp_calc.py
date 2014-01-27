import multiprocessing as mp
import time

class MPCompute(mp.Process):
    """
    Generic class for running computations that occur infrequently
    but take longer than a single BMI loop iteration
    """
    def __init__(self, work_queue, result_queue, fn):
        ''' __init__
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
        try:
            job = self.work_queue.get_nowait()
        except:
            job = None
        return job
        
    def run(self):
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
        return self.fn(*args, **kwargs)

    def stop(self):
        self.done.set()

class FuncProxy(object):
    def __init__(self, fn, multiproc=False, waiting_resp=None):
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

        self.prev_result = None
        self.prev_input = None
        self.waiting = False

    def _stuff(self):
        try:
            output_data = self.result_queue.get_nowait()
            self.prev_result = output_data
            self.waiting = False
            return output_data
        except Queue.Empty:
            if self.waiting_resp == None:
                return None
            elif self.waiting_resp == 'prev':
                return self.prev_result
        except:
            traceback.print_exc()

    def __call__(self, *args, **kwargs):
        if self.multiproc:
            input_data = (args, kwargs)
            input_same_as_last = input_data == self.prev_input

            if input_same_as_last and not self.waiting:
                return self.prev_result

            elif input_same_as_last and self.waiting:
                # Return the new result if it's available, otherwise the previous result
                return self._stuff()

            elif not input_same_as_last:
                self.work_queue.put(input_data)    
                self.prev_input = input_data
                self.waiting = True
                return self._stuff()
        else:
            return fn(*args, **kwargs)

    def __del__(self):
        # Stop the child process if one was spawned
        if self.multiproc:
            self.calculator.stop()