import numpy as np
import riglib.bmi

#block = 'cart20130428_01'
block = 'cart20130425_05'
files = dict(plexon='/storage/plexon/%s.plx' % block, hdf='/storage/rawdata/hdf/%s.hdf' % block)
binlen = 0.1
tslice = [1., 300.]

decoder = riglib.bmi.train._train_KFDecoder_manual_control(cells=None, binlen=0.1, tslice=[None,None],
    state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], **files) 
