#!/usr/bin/pyhon
from db import dbfunctions as dbfn
from tasks import performance
import matplotlib.pyplot as plt
import plotutil
import numpy as np

reload(performance)
class TaskEntrySet(object):
    def __init__(self, blocks):
        self.task_entries = map(performance._get_te, blocks)

    def map(self, fn):
        return np.array(map(fn, self.task_entries))

class BMIControlMultiTaskEntrySet(TaskEntrySet):
    @property
    def trials_per_min(self):
        return self.map(lambda te: te.trials_per_min)

    @property
    def perc_correct(self):
        return self.map(lambda te: te.perc_correct)

    @property
    def mean_ME(self):
        return self.map(lambda te: np.mean(te.ME))

    @property
    def mean_reach_times(self):
        return self.map(lambda te: np.mean(te.reach_times()))

        
fixed_blocks = [2185, 2187, 2189, 2191, 2193, 2195]
task_entry_set = BMIControlMultiTaskEntrySet(fixed_blocks)

#task_entries = map(performance._get_te, fixed_blocks)
#tau_values = map(lambda te: dbfn.TaskEntry(te.decoder_record.entry).params['tau'], task_entries)
#trials_per_min = map(lambda te: te.trials_per_min, task_entries)
#perc_correct = map(lambda te: te.perc_correct, task_entries)
#ME = map(lambda te: np.mean(te.ME), task_entries)
#reach_times = map(lambda te: np.mean(te.reach_times()), task_entries)

plt.close('all')
plt.figure()
axes = plotutil.subplots(3, 1, return_flat=True)
axes[0].plot(task_entry_set.trials_per_min)
axes[1].plot(task_entry_set.perc_correct)
axes[2].plot(task_entry_set.mean_ME)
axes[2].plot(task_entry_set.mean_reach_times)

plotutil.set_axlim(axes[0], axis='y')
plotutil.set_axlim(axes[1], axis='y')
plotutil.set_axlim(axes[2], axis='y')

plt.show()
