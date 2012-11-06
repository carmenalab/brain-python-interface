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

    def __init__(self, *args, **kwargs):
        super(MotionBMI, self).__init__(*args, **kwargs)
        rows = parse.rowbyte(self.plx.events[:])[0][:,0]

        self.kinem = tables.openFile(self.kinem).root.motiondata
        self.neurons = np.array([self.psth(self.plx[r-self.binlen*2:r]) for r in rows])
        assert len(self.kinem) == len(self.neurons)