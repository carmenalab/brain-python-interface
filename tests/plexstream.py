import time
from riglib import source
from riglib.plexon import Spikes
import matplotlib.pyplot as plt

ds = source.DataSource(Spikes, addr=('192.168.0.6', 6000))
ds.start()
time.sleep(1)
data = ds.get(True)
ds.stop()

plt.figure()
plt.plot(data['ts'])
plt.show()
