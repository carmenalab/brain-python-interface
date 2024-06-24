#!/usr/bin/python
'''
Test that the content of the npz file is the same as what is stored in the HDF file
'''
from db import dbfunctions as dbfn
from db.tracker import models
from tasks import performance
import plotutil
import tables
import numpy as np

te = performance._get_te(2440)
print(te)
te.clda_param_hist;
error = np.zeros(len(te.clda_param_hist))

for k, update in enumerate(te.clda_param_hist):
    if update is not None:
        print(k)
        error[k] = np.max(np.abs(te.hdf.root.clda[k]['filt_C'] - update['filt.C']))
print(np.max(np.abs(error)))
