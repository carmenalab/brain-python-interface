import time
from riglib import source
from riglib.plexon import Spikes

ds = source.DataSource(Spikes)
ds.start()
time.sleep(10)
data = ds.get(True)
ds.stop()
