from scipy.interpolate import Rbf
from riglib.experiment import traits, features

class ThinPlate(object):
    '''Interpolates arbitrary input dimensions into arbitrary output dimensions using thin plate splines'''
    def __init__(self, data, actual, smooth=0):
        self.data = data
        self.actual = actual
        self.smooth = smooth

        self.funcs = []
        for a in actual.T:
            f = Rbf(*np.vstack([data.T, a]), function='thin_plate', smooth=smooth)
            self.funcs.append(f)

    def __call__(self, data):
        return np.vstack([func(d) for func, d in zip(self.funcs, data.T)]).T

class Affine(object):
    '''Runs a linear affine interpolation between data and actual'''
    def __init__(self, data, actual):
        self.data = data
        self.actual = actual
        #self.xfm = np.linalg.lstsq()