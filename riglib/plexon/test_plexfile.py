import numpy as np
from plexon import plexfile

filename = "cart20130620_03.plx"
def test_continuous_edges():
	plx = plexfile.openFile(filename)
	lfp = plx.lfp[:10]
	data = lfp.data
	time = lfp.time
	assert np.allclose(data[1000:2000], plx.lfp[1:2].data)
	assert np.allclose(data[1000:2000], plx.lfp[time[1000]:time[2000]].data)
	assert np.allclose(data[1000:2001], plx.lfp[time[1000]:time[2000]+.00001].data)
	assert np.allclose(data[480:1080], plx.lfp[time[480]:time[1080]].data)
	assert np.allclose(data[479:1080], plx.lfp[time[479]:time[1080]].data)
	assert np.allclose(data[479:1079], plx.lfp[time[479]:time[1079]].data)
	assert np.allclose(data[480:1079], plx.lfp[time[480]:time[1079]].data)