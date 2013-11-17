#!/usr/bin/python
from db import dbfunctions as dbfn
from tasks import bmimultitasks, performance
import numpy as np

idx = 2298
te = performance._get_te(idx)
print te
hdf = te.hdf
dec = te.decoder
bmi_params = te.clda_param_hist

assist_level = hdf.root.task[:]['assist_level'].ravel()
spike_counts = hdf.root.task[:]['spike_counts']
target = hdf.root.task[:]['target']
cursor = hdf.root.task[:]['cursor']

task_msgs = hdf.root.task_msgs[:]
update_bmi_msgs = filter(lambda x: x['msg'] == 'update_bmi', task_msgs)
inds = [x[1] for x in update_bmi_msgs]

T = spike_counts.shape[0]
error = np.zeros(T)

for k in range(T):
    if k - 1 in inds:
        dec.update_params(bmi_params[inds.index(k-1)], steady_state=True)
    st = dec(spike_counts[k], target=target[k], target_radius=te.target_radius, assist_level=assist_level[k], speed=0.5)

    if cursor.dtype == np.float32:
        error[k] = np.linalg.norm(cursor[k] - np.float32(st[0:3, -1]))
    elif cursor.dtype == np.float64:
        error[k] = np.linalg.norm(cursor[k] - st[0:3, -1])
    else:
        raise TypeError("Cursor dtype unrecongized: %s" % cursor.dtype)

    if error[k] > 0:
        pass
        #import pdb; pdb.set_trace()

print 'Max error', np.max(np.abs(error))
