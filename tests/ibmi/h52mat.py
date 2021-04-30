import h5py
import sys
import scipy.io as sio

hdffile = sys.argv[1]

data = dict()

hdf = h5py.File(hdffile, 'r')
keys = list(hdf.get('channel').keys())
for key in keys:
	if 'channel' in key:
		chan = key[-5:]
		samples = hdf.get('channel/%s/continuous_set' % key).value
		data[chan] = samples

save_dict = dict()
for chan in data:
    save_dict['chan' + str(chan).zfill(5)] = data[chan]

sio.matlab.savemat(hdffile[:-4] + '.mat', save_dict)
