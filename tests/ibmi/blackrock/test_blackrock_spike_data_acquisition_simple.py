import time
from riglib import blackrock

channels = [5, 6, 7, 8]
obj = blackrock.Spikes(channels)

obj.start()

for i in range(10):
    print(obj.get())
    time.sleep(0.01)
