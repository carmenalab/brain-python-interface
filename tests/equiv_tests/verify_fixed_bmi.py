#!/usr/bin/python
'''
Test case to check that the current state of the code is able to reconstruct a 
TaskEntry using the BMIControlMulti task 
'''
import tasks
from tasks import performance
from tasks import generatorfunctions as genfns
import numpy as np

idx = 2295
te = performance._get_te(idx)
n_iter = len(te.hdf.root.task)

gen = genfns.sim_target_seq_generator_multi(8, 1000)
self = task = tasks.BMIReconstruction(te, n_iter, gen)
task.init()

error = task.calc_recon_error(verbose=True)
print "Recon error", np.max(np.abs(error))
