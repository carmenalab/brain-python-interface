import time
from riglib import brainamp

channels = [1]
obj = brainamp.EMG(channels)

obj.start()

for i in range(10):
    print obj.get()
    time.sleep(0.01)
