import time
import numpy as np
import tempfile
import glob
np.set_printoptions(suppress=True)
from riglib.sink import sinks, PrintSink
from riglib import source
from riglib import motiontracker, hdfwriter, nidaq

Motion = motiontracker.make(38, motiontracker.System)
datasource = source.DataSource(Motion)
tf = tempfile.NamedTemporaryFile()
sinks.start(hdfwriter.HDFWriter, filename=tf.name)
sinks.register(Motion)


datasource.start()
time.sleep(10)
datasource.stop()
sinks.stop()
time.sleep(2)

import tables
h5 = tables.openFile(tf.name)
caldat = h5.root.motiontracker[1024:, -6:,:3]

d = glob.glob("/storage/calibrations/alignment*.npz")
motiontracker.make_autoalign_reference(caldat, "/storage/calibrations/alignment%d.npz"%len(d))
