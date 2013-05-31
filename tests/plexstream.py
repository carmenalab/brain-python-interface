import time
from riglib import source
from riglib.plexon import Spikes
import matplotlib.pyplot as plt

ds = source.DataSource(Spikes, addr=('10.0.0.13', 6000))
ds.start()
time.sleep(10)
data = ds.get(True)
ds.stop()

plt.figure()
plt.plot(data['ts'])
plt.show()
