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

class DataFilter(features.DataSource, traits.HasTraits):
    '''A feature which gets data from the type named and applies the calibration profile provided'''
    profiles = traits.List(trait=traits.Instance('models.Calibration'))

    def __init__(self, *args, **kwargs):
        super(DataFilter, *args, **kwargs)
        self._datasource = self.datasource

        datasource = self.datasource
        calibrations = dict((p.name, p.get()) for profile in self.profiles)

        class Proxy(object):
            def get(self, mode):
                return self.calibrations[mode]()