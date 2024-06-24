#!/usr/bin/python
"""
Test that training a KFDecoder from a BMI file works
(syntax check, not functionality)
"""
import riglib.bmi
block = 'cart20130521_04'
files = dict(hdf='/storage/rawdata/hdf/%s.hdf' % block)
binlen = 0.1
tslice = [1., 300.]

decoder = riglib.bmi.train._train_KFDecoder_brain_control(cells=None, binlen=0.1, tslice=[None,None],
    state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], **files)
