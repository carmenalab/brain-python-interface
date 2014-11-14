import time
import threading


class Counter(threading.Thread):
    def __init__(self, count, t_step):
        super(Counter, self).__init__()
        self.count = 0
        self.t_step = t_step
        self.active = True
        self.lock = threading.Lock()

    def run(self):
        print 'starting!'
        self.active = True
        t_start = time.time()
        t_elapsed = 0
        t_simulated = 0
        
        while self.active:
            t_elapsed = time.time() - t_start
            if t_elapsed > t_simulated:
                tmp = self.count
                self.count = 2000
                self.count = tmp + self.t_step
                # with self.lock:
                #     self.count += self.t_step
    
                t_simulated += self.t_step
                x = [i for i in range(5000)]  # takes about 0.0005 s (0.5 ms)

    def poll(self):
        # with self.lock:
        #     return self.count
        return self.count

    def stop(self):
        self.active = 0

    def set_count(self, count):
        with self.lock:
            self.count = count
    

if __name__ == '__main__':
    c = Counter(0, 0.001)
    c.start()

    # time.sleep(3)
    t_start = time.time()
    for i in xrange(4400000):
        j = 1
    print 't_elapsed:', time.time() - t_start
    print 'Count (should be ~= 3):', c.poll()

    c.set_count(10)  # set count to 10
    time.sleep(2)
    print 'Count (should be ~= 12):', c.poll()

    # time.sleep(20)
    # print c.poll()

    c.stop()

