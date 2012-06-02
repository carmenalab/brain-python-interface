import numpy as np
from scipy.interpolate import Rbf

class Profile(object):
    def __init__(self, data, actual, system=None):
        self.data = np.array(data)
        self.actual = np.array(actual)
        self.system = system
    
    def _init(self):
        #Sanitize the data, clearing out entries which are invalid
        pass

class EyeProfile(Profile):
    def __init__(self, data, actual, **kwargs):
        super(EyeProfile, self).__init__(data, actual, system="eyetracker", **kwargs)
    
    def _init(self):
        valid = -(self.data == (-32768, -32768)).all(1)
        self.data = self.data[valid,:]
        self.actual = self.actual[valid,:]
        super(EyeProfile, self)._init()

class ThinPlate(Profile):
    '''Interpolates arbitrary input dimensions into arbitrary output dimensions using thin plate splines'''
    def __init__(self, data, actual, smooth=0, **kwargs):
        super(ThinPlate, self).__init__(data, actual, **kwargs)
        self.smooth = smooth
        self._init()
    
    def _init(self):
        super(ThinPlate, self)._init()
        self.funcs = []
        for a in self.actual.T:
            f = Rbf(*np.vstack([self.data.T, a]), function='thin_plate', smooth=self.smooth)
            self.funcs.append(f)
        
    def __call__(self, data):
        return np.array([func(d) for func, d in zip(self.funcs, np.array(data).T)]).T
    
    def __getstate__(self):
        return dict(data=self.data, actual=self.actual, smooth=self.smooth)
    
    def __setstate__(self, state):
        super(ThinPlate, self).__setstate__(state)
        self._init()

class ThinPlateEye(EyeProfile, ThinPlate):
    pass

def crossval(cls, data, actual, proportion=0.7, parameter="smooth", 
    xval_range=np.linspace(0,5,20)**2):
    actual = np.array(actual)
    data = np.array(data)

    idx = np.random.permutation(len(actual))
    border = int(proportion*len(actual))
    trn, val = idx[:border], idx[border:]

    dim = tuple(range(data.shape[1]/2)), tuple(range(data.shape[1]/2, data.shape[1]))
    ccs = np.zeros(len(xval_range))
    for i, smooth in enumerate(xval_range):
        cal = cls(data[trn], actual[trn], **{parameter:smooth})
        ccdata = np.hstack([cal(data[val]), actual[val]]).T
        ccs[i] = np.corrcoef(ccdata)[dim].mean()
    
    best = xval_range[ccs.argmax()]
    return cls(data, actual, **{parameter:best}), best, ccs

class Affine(Profile):
    '''Runs a linear affine interpolation between data and actual'''
    def __init__(self, data, actual):
        self.data = data
        self.actual = actual
        #self.xfm = np.linalg.lstsq()
