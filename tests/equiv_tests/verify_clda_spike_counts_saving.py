# coding: utf-8
from db import dbfunctions as dbfn
import numpy as np

task_entry = 2023

dec = dbfn.get_decoder(task_entry)

try:
    dec.bminum
except:
    dec.bminum = 1


bmi_params = np.load(dbfn.get_bmiparams_file(task_entry))
hdf = dbfn.get_hdf(task_entry)
task_msgs = hdf.root.task_msgs[:]
update_bmi_msgs = [msg for msg in task_msgs if msg['msg'] == 'update_bmi']
spike_counts = hdf.root.task[:]['spike_counts']

for k, msg in enumerate(update_bmi_msgs):
    time = msg['time']
    hdf_sc = np.sum(spike_counts[time-dec.bminum+1:time+1], axis=0)
    if not np.array_equal(hdf_sc, bmi_params[k]['spike_counts_batch']):
        print(k)
