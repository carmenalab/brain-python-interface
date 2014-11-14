import os
import numpy as np
from riglib.nidaq import parse
import subprocess

def sys_eq(sys1, sys2):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    return sys1 in [sys2, sys2[1:]]



nev_fname = '/storage/blackrock/20140707-171012/20140707-171012-002.nev'
tslice = [3, 13]
sys_name = 'task'

nev_hdf_fname = nev_fname + '.hdf'
if not os.path.isfile(nev_hdf_fname):
    # convert .nev file to hdf file using Blackrock's n2h5 utility
    subprocess.call(['n2h5', nev_fname, nev_hdf_fname])

import h5py
nev_hdf = h5py.File(nev_hdf_fname, 'r')

path = 'channel/digital00001/digital_set'
ts = nev_hdf.get(path).value['TimeStamp']
msgs = nev_hdf.get(path).value['Value']

# copied from riglib/nidaq/parse.py
msgtype_mask = 0b0000111<<8
auxdata_mask = 0b1111000<<8
rawdata_mask = 0b11111111
msgtype = np.right_shift(np.bitwise_and(msgs, msgtype_mask), 8).astype(np.uint8)
# auxdata = np.right_shift(np.bitwise_and(msgs, auxdata_mask), 8).astype(np.uint8)
auxdata = np.right_shift(np.bitwise_and(msgs, auxdata_mask), 11).astype(np.uint8)
rawdata = np.bitwise_and(msgs, rawdata_mask)

# data is an N x 4 matrix that will be the argument to parse.registrations()
data = np.vstack([ts, msgtype, auxdata, rawdata]).T

# print 'data', data

# get system registrations
reg = parse.registrations(data)

# import pprint
# pprint.pprint(reg.items())
# print 'reg.items', reg.items()
# print 'len(reg.items())', len(reg.items())

syskey = None

for key, system in reg.items():
    if sys_eq(system[0], sys_name):
        syskey = key
        break

if syskey is None:
    raise Exception('No source registration saved in the file!')

# print 'parse.rowbyte(data)', parse.rowbyte(data)

# get the corresponding hdf rows
rows = parse.rowbyte(data)[syskey][:,0]

timestamps = rows / 30000.

lower, upper = 0 < timestamps, timestamps < timestamps.max() + 1
l, u = tslice
if l is not None:
    lower = l < timestamps
if u is not None:
    upper = timestamps < u
tmask = np.logical_and(lower, upper)
