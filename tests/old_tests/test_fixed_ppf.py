#!/usr/bin/python
'''
Test case to check that the current state of the code is able to reconstruct a 
TaskEntry using the BMIControlMulti task 
'''
from analysis import performance
from tasks import bmi_recon_tasks
import numpy as np

idx = 2295
te = performance.BMIControlMultiTaskEntry(idx, dbname='testing')
n_iter = len(te.hdf.root.task)

gen = bmi_recon_tasks.FixedPPFBMIReconstruction.centerout_2D_discrete()
self = task = bmi_recon_tasks.FixedPPFBMIReconstruction(te, n_iter, gen)
task.init()

error = task.calc_recon_error(verbose=False, n_iter_betw_fb=1000)
print("Recon error", np.max(np.abs(error)))
