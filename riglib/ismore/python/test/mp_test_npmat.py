import time
import multiprocessing as mp
import numpy as np


def shared_np_mat(dtype, shape):
    tmp = mp.RawArray('c', dtype.itemsize * shape[0] * shape[1])
    return np.mat(np.frombuffer(tmp, dtype).reshape(shape))



class Counter(mp.Process):
    def __init__(self, count, t_step):
        super(Counter, self).__init__()
        dtype = np.dtype('float64')
        self.count_mat = shared_np_mat(dtype, (2, 2))
        self.t_step = t_step
        self.active = mp.Value('b', 0)
        self.lock = mp.Lock()

    def run(self):
        print 'starting!'
        self.active.value = 1
        t_start = time.time()
        t_elapsed = 0
        t_simulated = 0
        
        while self.active.value == 1:
            t_elapsed = time.time() - t_start
            if t_elapsed > t_simulated:
                self.lock.acquire()
                self.count_mat += self.t_step
                self.lock.release()

                t_simulated += self.t_step

    def poll(self):
        return self.count_mat

    def stop(self):
        self.active.value = 0

    def set_count(self, count):
        self.lock.acquire()
        self.count_mat[:, :] = count
        self.lock.release()


if __name__ == '__main__':
    c = Counter(0, 0.0001)
    c.start()

    time.sleep(3)
    print 'Count (should be ~= 3):', c.poll()

    c.set_count(10)  # set count to 10
    time.sleep(2)
    print 'Count (should be ~= 12):', c.poll()

    c.stop()

