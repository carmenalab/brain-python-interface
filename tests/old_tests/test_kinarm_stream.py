import time
import numpy as np
from riglib.sink import sinks, PrintSink
from riglib import source
from riglib import kinarmdata, hdfwriter

Motion = kinarmdata.Kinarmdata
datasource = source.DataSource(Motion)

sinks.start(hdfwriter.HDFWriter, filename="/tmp/test.hdf")
#sinks.start(PrintSink)
sinks.register(Motion)

datasource.start()
print("reading for 10 seconds...")
t0 = time.time()
try:
    while (time.time()-t0)<10:
        g= datasource.get()
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
datasource.stop()

sinks.stop()
print("complete!")
