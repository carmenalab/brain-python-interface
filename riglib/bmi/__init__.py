import numpy as np
import tables

from riglib.plexon import plexfile, psth
from riglib.nidaq import parse

class BMI(object):
    '''A BMI object, for filtering neural data into BMI output'''

    def __init__(self, cells, binlen=.1, **kwargs):
        '''All BMI objects must be pickleable objects with two main methods -- the init method trains
        the decoder given the inputs, and the call method actually does real time filtering'''
        self.files = kwargs
        self.binlen = binlen

        self.psth = psth.Filter(cells, binlen)

    def __call__(self, data):
        psth = self.psth(data)
        return psth

    def __setstate__(self, state):
        self.psth = psth.Filter(state['cells'], state['binlen'])
        del state['cells']
        self.__dict__.update(state)

    def __getstate__(self):
        state = dict(cells=self.psth.chans)
        exclude = set(['plx', 'psth'])
        for k, v in self.__dict__.items():
            if k not in exclude:
                state[k] = v
        return state

    def get_data(self):
        raise NotImplementedError

class MotionBMI(BMI):
    '''BMI object which is trained from motion data'''

    def __init__(self, cells, tslice=(None, None), **kwargs):
        super(MotionBMI, self).__init__(cells, **kwargs)
        assert 'hdf' in self.files and 'plexon' in self.files
        self.tslice = tslice

    def get_data(self):
        plx = plexfile.openFile(self.files['plexon'])
        rows = parse.rowbyte(plx.events[:])[0][:,0]
        lower, upper = 0 < rows, rows < rows.max()+1
        l, u = self.tslice
        if l is not None:
            lower = l < rows
        if u is not None:
            upper = rows < u
        mask = np.logical_and(lower, upper)

        #Trim the mask to have exactly an even multiple of 4 worth of data
        if len(mask) % 4 != 0:
            mask[-(len(mask) % 4):] = False

        #Grab masked data, filter out interpolated data
        motion = tables.openFile(self.files['hdf']).root.motiontracker
        t, m, d = motion.shape
        motion = motion[np.tile(mask, [d,m,1]).T].reshape(-1, 4, m, d)
        invalid = np.logical_and(motion[...,-1] == 4, motion[..., -1] < 0)
        motion[invalid] = 0
        
        kin = motion.sum(1)
        neurons = np.array([self.psth(plx.spikes[r-self.binlen*2:r]) for r in rows[mask][3::4]])
        assert len(kin) == len(neurons)
        return kin, neurons

class ManualBMI(MotionBMI):
    def __init__(self, states=['origin_hold', 'terminus', 'terminus_hold', 'reward'], *args, **kwargs):
        super(ManualBMI, self).__init__(*args, **kwargs)
        self.states = states

    def get_data(self):
        h5 = tables.openFile(self.files['hdf'])
        states = h5.root.motiontracker_msgs[:]
        names = dict((n, i) for i, n in enumerate(np.unique(states['msg'])))
        target = np.array([names[n] for n in self.states])
        seq = np.array([(names[n], t) for n, t, in states])

        idx = np.convolve(target, target, 'valid')
        found = np.convolve(seq[:,0], target, 'valid') == idx

        slices = states[found]['time'].reshape(-1, 2)
        t, m, d = h5.root.motiontracker.shape
        mask = np.ones((t/4, m, d), dtype=bool)
        for s, e in slices:
            mask[s/4:e/4] = False
        kin, neurons = super(ManualBMI, self).get_data()
        return np.ma.array(kin, mask=mask), neurons

class VelocityBMI(ManualBMI):
    def get_data(self):
        kin, neurons = super(VelocityBMI, self).get_data()
        velocity = np.diff(kin[...,:3], axis=0)
        kin = np.hstack([kin[1:,:,:3], velocity])
        return kin, neurons[1:]

from kalman import KalmanFilter
