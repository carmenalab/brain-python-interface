'''
Test parameter reconstruction from the saved CLDA parameters
'''
import unittest
import tasks
from analysis import performance
from tasks import generatorfunctions as genfns
import numpy as np 

from tasks import bmi_recon_tasks
from riglib.bmi import clda
reload(bmi_recon_tasks)
reload(tasks)
    

cls=tasks.KFRMLCGRecon

idx = 5275

te = performance._get_te(idx)

### TODO some CLDA blocks have a changing half life...
updater = clda.KFRML(None, None, te.batch_time, te.half_life[0])
updater.init(te.seed_decoder)

param_hist = te.hdf.root.clda
C_error = []
Q_error = []
decoder = te.seed_decoder

for k in range(len(param_hist))[5::6]:
    intended_kin = param_hist[k]['intended_kin']

    spike_counts = param_hist[k]['spike_counts_batch']
    if not np.any(np.isnan(intended_kin)) and not np.any(np.isnan(spike_counts)):
        new_params = updater.calc(intended_kin, spike_counts, decoder)
        decoder.update_params(new_params)
        C_error.append(np.max(np.abs(decoder.filt.C - param_hist[k]['kf_C'])))
        Q_error.append(np.max(np.abs(decoder.filt.Q - param_hist[k]['kf_Q'])))
    if (k - 5)/6 % 100 == 0: print(k)


print('max C err', np.max(C_error))
print('max Q err', np.max(Q_error))