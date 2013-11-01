# coding: utf-8
from db import dbfunctions as dbfn
import numpy as np

task_entry = 2015
bmi_params = np.load(dbfn.get_bmiparams_file(task_entry))
hdf = dbfn.get_hdf(task_entry)
task_msgs = hdf.root.task_msgs[:]
update_bmi_msgs = filter(lambda msg: msg['msg'] == 'update_bmi', task_msgs)
spike_counts = hdf.root.task[:]['spike_counts']
for k, msg in enumerate(update_bmi_msgs):
    assert np.array_equal(spike_counts[msg['time']], bmi_params[k]['spike_counts_batch'])
