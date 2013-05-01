import time
from riglib import source
from riglib.plexon import SimSpikes

ds = source.DataSource(SimSpikes)
ds.start()
time.sleep(10)
data = ds.get(True)
ds.stop()
