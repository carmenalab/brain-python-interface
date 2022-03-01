import subprocess
import os
import h5py
import numpy as np

nsx_fs = dict()
nsx_fs['.ns1'] = 500
nsx_fs['.ns2'] = 1000
nsx_fs['.ns3'] = 2000
nsx_fs['.ns4'] = 10000
nsx_fs['.ns5'] = 30000
nsx_fs['.ns6'] = 30000

NSP_channels = np.arange(128) + 1

# files = [u'/storage/blackrock/20140709-161439/20140709-161439-003.ns3']
from db.tracker import models
files = models.TaskEntry.objects.get(pk=62).get_nsx_files

lengths = []
for nsx_fname in files:

	nsx_hdf_fname = nsx_fname + '.hdf'
	if not os.path.isfile(nsx_hdf_fname):
	    # convert .nsx file to hdf file using Blackrock's n2h5 utility
	    subprocess.call(['n2h5', nsx_fname, nsx_hdf_fname])

	nsx_hdf = h5py.File(nsx_hdf_fname, 'r')

	for chan in NSP_channels:
		chan_str = str(chan).zfill(5)
		path = 'channel/channel%s/continuous_set' % chan_str
		if nsx_hdf.get(path) is not None:
			last_ts = len(nsx_hdf.get(path).value)
			fs = nsx_fs[nsx_fname[-4:]]

			length = last_ts / fs
			lengths.append(length)
			print(length)

			break

print(lengths)
print(max(lengths))

