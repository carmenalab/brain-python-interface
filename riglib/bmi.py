import numpy as np
import tables

from riglib.plexon import plexfile, psth
from riglib.nidaq import parse

class BMI(object):
    '''A BMI object, for filtering neural data into BMI output'''

    def __init__(self, kinem, plxfile, cells, binlen=.1):
        '''All BMI objects must be pickleable objects with two main methods -- the init method trains
        the decoder given the inputs, and the call method actually does real time filtering'''
        self.kinem = kinem
        self.binlen = binlen
        self.plx = plexfile.openFile(plxfile)
        self.psth = psth.Filter(cells, binlen)

    def __call__(self, data):
        return self.psth(data)

class MotionBMI(BMI):
    '''BMI object which is trained from motion data'''

    def __init__(self, tslice=(None, None), *args, **kwargs):
        super(MotionBMI, self).__init__(*args, **kwargs)

        rows = parse.rowbyte(self.plx.events[:])[0][:,0]
        lower, upper = 0 < rows, rows < rows.max()
        if tslice[0] is not None:
            lower = tslice[0] < rows
        if tslice[1] is not None:
            upper = rows < tslice[1]
        mask = np.logical_and(lower, upper)

        #Trim the mask to have exactly an even multiple of 4 worth of data
        midx, = np.nonzero(mask)
        mask[midx[-(sum(mask)%4):]] = False

        #Grab masked data, filter out interpolated data
        motion = tables.openFile(self.kinem).root.motiontracker
        t, m, d = motion.shape
        motion = motion[np.tile(mask, [d,m,1]).T].reshape(-1, 4, m, d)
        invalid = np.logical_and(motion[...,-1] == 4, motion[..., -1] < 0)
        motion[invalid] = 0
        
        self.kinem = motion.sum(1)
        self.neurons = np.array([self.psth(self.plx.spikes[r-self.binlen*2:r]) for r in rows[mask][3::4]])

        assert len(self.kinem) == len(self.neurons)