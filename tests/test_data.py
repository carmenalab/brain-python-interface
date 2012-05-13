import time

from riglib.sink import sinks
from riglib import source
from riglib import motiontracker, hdfwriter

Motion = motiontracker.make_simulate(8)
datasource = source.DataSource(Motion)
sinks.start(hdfwriter.HDFWriter, filename="/tmp/test.hdf")
sinks.register(Motion)

datasource.start()
print "reading for 10 seconds..."
time.sleep(10)
datasource.stop()

sinks.stop()
print "complete!"