import time
import multiprocessing as mp


class Counter(mp.Process):
    def __init__(self, count, t_step):
        super(Counter, self).__init__()
        self.count = mp.Value('f', 0)
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
            if t_elapsed - t_simulated > self.t_step:
                self.lock.acquire()
                self.count.value += self.t_step
                self.lock.release()

                t_simulated += self.t_step
                for i in range(5000):
                    pass

    def poll(self):
        return self.count.value

    def stop(self):
        self.active.value = 0

    def set_count(self, count):
        self.lock.acquire()
        self.count.value = count
        self.lock.release()


if __name__ == '__main__':
    c = Counter(0, 0.001)
    c.start()

    time.sleep(3)

    t_start = time.time()
    for i in xrange(4400000):
        j = 1
    print 't_elapsed:', time.time() - t_start

    print 'Count (should be ~= 3):', c.poll()

    c.set_count(10)  # set count to 10
    time.sleep(2)
    print 'Count (should be ~= 12):', c.poll()

    c.stop()

