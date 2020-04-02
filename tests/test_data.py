import time
import numpy as np
np.set_printoptions(suppress=True)
from riglib.sink import sinks, PrintSink
from riglib import source
from riglib import motiontracker, hdfwriter, nidaq, calibrations

Motion = motiontracker.make(8, motiontracker.AligningSystem)

datasource = source.DataSource(Motion)
datasource.filter = calibrations.AutoAlign()
#sinks.start(nidaq.SendRowByte)
#sinks.register(Motion)

sinks.start(hdfwriter.HDFWriter, filename="/tmp/test.hdf")
#sinks.start(PrintSink)
sinks.register(Motion)

datasource.start()
print("reading for 10 seconds...")
try:
    while True:
        print(datasource.get())
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
datasource.stop()

sinks.stop()
print("complete!")
