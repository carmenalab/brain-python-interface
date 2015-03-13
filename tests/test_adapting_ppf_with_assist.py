import unittest
import tasks
from analysis import performance
from tasks import generatorfunctions as genfns
import numpy as np 

from tasks import bmi_recon_tasks
from tasks.bmimultitasks import SimBMIControlMulti

reload(bmi_recon_tasks)
reload(tasks)
    

cls = bmi_recon_tasks.ContCLDARecon
idx = 2554

te = performance.CLDAControlMultiTaskEntry(idx, dbname='testing')
n_iter = len(te.hdf.root.task)
        
gen = SimBMIControlMulti.sim_target_seq_generator_multi(8, 1000)
task = cls(te, n_iter, gen)
task.init()
        
error = task.calc_recon_error(verbose=False)
abs_max_error = np.max(np.abs(error))

print 'abs_max_error', abs_max_error

# ['intended_kin',
#  'kf.Q',
#  'sdFR',
#  'kf.C_xpose_Q_inv_C',
#  'mFR',
#  'kf.C',
#  'kf.ESS',
#  'filt.S',
#  'filt.R',
#  'filt.T',
#  'kf.C_xpose_Q_inv',
#  'spike_counts_batch']
