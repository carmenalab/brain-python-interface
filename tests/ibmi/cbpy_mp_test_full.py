import time
import sys
import multiprocessing as mp

use_new_cbpy = True

if use_new_cbpy:
    from cerebus import cbpy

    class DedicatedCbpyProcess(mp.Process):
        def __init__(self):
            super(DedicatedCbpyProcess, self).__init__()

            self.parameters = dict()
            self.parameters['inst-addr']   = '192.168.137.128'
            self.parameters['inst-port']   = 51001
            self.parameters['client-port'] = 51002

            if sys.platform == 'darwin':  # OS X
                print 'Using OS X settings for cbpy.open'
                self.parameters['client-addr'] = '255.255.255.255'
            else:  # linux
                self.parameters['client-addr'] = '192.168.137.255'
                self.parameters['receive-buffer-size'] = 8388608  # necessary?

        def run(self):
            print 'calling cbpy.open'
            print 'self.parameters:', self.parameters
            result, return_dict = cbpy.open(connection='default', parameter=self.parameters)
            print 'cbpy.open result:', result
            print 'cbpy.return_dict:', return_dict

            range_parameter = dict()
            range_parameter['begin_channel'] = 5
            range_parameter['end_channel']   = 8

            result, reset = cbpy.trial_config(range_parameter=range_parameter)
            print 'result:', result
            print 'reset:', reset
            print ''

            n_secs = 20
            loop_time = 0.1
            n_itrs = int(n_secs / loop_time)

            for itr in range(n_itrs):
                t_start = time.time()
                print '\nitr %d of %d:' % (itr+1, n_itrs)

                print 'calling cbpy.trial_event()'
                result, trial = cbpy.trial_event()
                # print 'calling cbpy.trial_continuous()'
                # result, trial = cbpy.trial_continuous()
                print 'result:', result
                print 'trial:', trial
                print ''

                t_elapsed = time.time() - t_start
                if t_elapsed < loop_time:
                    time.sleep(loop_time - t_elapsed)
                print 't_elapsed: %3.3f secs, loop time: %3.3f secs' % (t_elapsed, time.time()-t_start)


            result = cbpy.close()
            print 'result:', result
            print ''


    p1 = DedicatedCbpyProcess()
    p1.start()

else:  # use "old cbpy" (CereLink.cbpy)
    from CereLink import cbpy

    class DedicatedCbpyProcess(mp.Process):
        def __init__(self):
            super(DedicatedCbpyProcess, self).__init__()

            self.parameters = dict()
            self.parameters['inst-addr']   = '192.168.137.128'
            self.parameters['inst-port']   = 51001
            self.parameters['client-port'] = 51002

            if sys.platform == 'darwin':  # OS X
                print 'Using OS X settings for cbpy.open'
                self.parameters['client-addr'] = '255.255.255.255'
            else:  # linux
                self.parameters['client-addr'] = '192.168.137.255'


        def run(self):
            print 'calling cbpy.open'
            print 'self.parameters:', self.parameters
            return_dict = cbpy.open('default', self.parameters)
            print 'connection:', return_dict['connection']
            print 'instrument:', return_dict['instrument']

    p1 = DedicatedCbpyProcess()
    p1.start()
