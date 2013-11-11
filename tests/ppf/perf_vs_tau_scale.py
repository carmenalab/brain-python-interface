#!/usr/bin/pyhon
from db import dbfunctions as dbfn
from tasks import performance
import matplotlib.pyplot as plt
import plotutil
import numpy as np

fixed_blocks = [2185, 2187, 2189, 2191, 2193, 2195]
task_entries = map(performance._get_te, fixed_blocks)
tau_values = map(lambda te: dbfn.TaskEntry(te.decoder_record.entry).params['tau'], task_entries)
trials_per_min = map(lambda te: te.trials_per_min, task_entries)
perc_correct = map(lambda te: te.perc_correct, task_entries)
ME = map(lambda te: np.mean(te.ME), task_entries)

plt.close('all')
plt.figure()
axes = plotutil.subplots(3, 1, return_flat=True)
axes[0].plot(tau_values, trials_per_min)
axes[1].plot(tau_values, perc_correct)
axes[2].plot(tau_values, ME)

plotutil.set_xlim(axes[0], [min(tau_values), max(tau_values)])
plotutil.set_xlim(axes[1], [min(tau_values), max(tau_values)])
plotutil.set_xlim(axes[2], [min(tau_values), max(tau_values)])
plotutil.set_axlim(axes[0], axis='y')
plotutil.set_axlim(axes[1], axis='y')
plotutil.set_axlim(axes[2], axis='y')

plt.show()
