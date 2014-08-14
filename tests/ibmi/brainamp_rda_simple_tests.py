# test 1

import time
from riglib import source, brainamp

channels = [1]
obj = brainamp.EMG('192.168.137.1')

obj.start()

for i in range(10):
    print obj.get()
    time.sleep(0.01)


# test 2

channels = [1]
s = source.MultiChanDataSource(brainamp.EMG, channels=channels, recorder_ip='192.168.137.1')
s.start()
