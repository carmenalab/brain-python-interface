import multiprocessing as mp

class MPCompute(mp.Process):
    """
    Generic class for CLDA parameter recomputation
    """
    def __init__(self, work_queue, result_queue):
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

    def _check_for_job(self):
        try:
            job = self.work_queue.get_nowait()
        except:
            job = None
        return job
        
    def run(self):
        """ The main loop """
        while not self.done.is_set():
            job = self._check_for_job()

            # unpack the data
            if not job == None:
                new_params = self.calc(*job)
                self.result_queue.put(new_params)

            # Pause to lower the process's effective priority
            time.sleep(0.5)

    def calc(self, *args, **kwargs):
        """Re-calculate parameters based on input arguments.  This
        method should be overwritten for any useful CLDA to occur!"""
        return None

    def stop(self):
        self.done.set()

class FuncProxy(object):
    def __init__(self, fn, multiproc=False, waiting_resp=None):
        self.multiproc = multiproc
        if self.multiproc:
            # create the class
            class Proc(MPCompute):
                def calc(self, *args, **kwargs):
                    fn(*args, **kwargs)

            # create the queues

            # spawn the process
        else:
            self.fn = fn

    def __call__(self, *args, **kwargs):
        if self.multiproc:
            pass
        else:
            return fn(*args, **kwargs)
