from plexon import plexfile
import numpy as np
plx = plexfile.openFile("/home/james/Downloads/cart20121106_04.plx")
from riglib.nidaq import parse
ts = parse.rowbyte(plx.events[:].data)[0]
print("binning...")
bins = np.array(list(plx.spikes.bin(ts[:,0])))
print("done!")