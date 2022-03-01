#!/usr/bin/python
from db import dbfunctions
from tasks import bmimultitasks
import numpy as np
from riglib.bmi import state_space_models, kfdecoder, train
reload(kfdecoder)

te = dbfunctions.get_task_entry(1883) # Block with predict and update of kf running at 10Hz
hdf = dbfunctions.get_hdf(te)
dec = dbfunctions.get_decoder(te)

assist_level = hdf.root.task[:]['assist_level'].ravel()
spike_counts = hdf.root.task[:]['spike_counts']
target = hdf.root.task[:]['target']
cursor = hdf.root.task[:]['cursor']

assert np.all(assist_level == 0)

task_msgs = hdf.root.task_msgs[:]
update_bmi_msgs = [x for x in task_msgs if x['msg'] == 'update_bmi']
inds = [x[1] for x in update_bmi_msgs]

assert len(inds) == 0

T = spike_counts.shape[0]
error = np.zeros(T)
for k in range(spike_counts.shape[0]):
    if k - 1 in inds:
        dec.update_params(bmi_params[inds.index(k-1)])
    st = dec(spike_counts[k], target=target[k], target_radius=1.8, assist_level=assist_level[k])

    error[k] = np.linalg.norm(cursor[k] - np.float32(st[0:3]))
    if error[k] > 1e-6:
        print("error!")
        break

print("Reconstruction error: ", np.max(np.abs(error)))

# Convert the SSM to 60Hz
dec_new = train._interpolate_KFDecoder_state_between_updates(dbfunctions.get_decoder(te))
error_with_interp = np.zeros(T)
st = np.zeros([T, 7])
for k in range(T):
    st[k] = dec_new(spike_counts[k], target=target[k], target_radius=1.8, assist_level=assist_level[k])
    error_with_interp[k] = np.linalg.norm(cursor[k] - np.float32(st[k, 0:3]))

print("Max error with interpolation: ", np.max(np.abs(error_with_interp[5::6]))) 
