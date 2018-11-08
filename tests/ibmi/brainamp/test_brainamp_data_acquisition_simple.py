import time
from riglib import source
from riglib.brainamp import rda

channels = [1]
obj = rda.EMGData('192.168.137.1')

obj.start()

for i in range(10):
    print("iteration i = " + str(i))
    print(obj.get())
    time.sleep(0.01)
