import time
from riglib import source
from riglib.plexon import Spikes

ds = source.DataSource(Spikes, addr=('10.0.0.13', 6000), channels=[22])
ds.start()
time.sleep(10)
data = ds.get(True)
ds.stop()

print(data)
