#!/usr/bin/python
from db import dbfunctions
from tasks import bmimultitasks
import numpy as np

te = dbfunctions.get_task_entry(1807)
hdf = dbfunctions.get_hdf(te)
dec = dbfunctions.get_decoder(te)
bmi_params = np.load(dbfunctions.get_bmiparams_file(te))

assist_level = hdf.root.task[:]['assist_level'].ravel()
spike_counts = hdf.root.task[:]['spike_counts']
target = hdf.root.task[:]['target']
cursor = hdf.root.task[:]['cursor']

task_msgs = hdf.root.task_msgs[:]
update_bmi_msgs = filter(lambda x: x['msg'] == 'update_bmi', task_msgs)
inds = [x[1] for x in update_bmi_msgs]

T = spike_counts.shape[0]
error = np.zeros(T)
for k in range(spike_counts.shape[0]):
    if k - 1 in inds:
        dec.update_params(bmi_params[inds.index(k-1)])
    st = dec(spike_counts[k], target=target[k], target_radius=1.8, assist_level=assist_level[k])

    error[k] = np.linalg.norm(cursor[k] - np.float32(st[0:3]))
    if error[k] > 1e-6:
        break
