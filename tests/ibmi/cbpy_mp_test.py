import sys
import multiprocessing as mp

use_new_cbpy = False

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
