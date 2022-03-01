'''
Calibration for the EyeLink eyetracker
'''

import numpy as np

class Profile(object):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, data, actual, system=None, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.data = np.array(data)
        self.actual = np.array(actual)
        self.system = system
        self.kwargs = kwargs
        self._init()
    
    def _init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        #Sanitize the data, clearing out entries which are invalid
        valid = ~np.isnan(self.data).any(1)
        self.data = self.data[valid,:]
        self.actual = self.actual[valid,:]

    def performance(self, blocks=5):
        '''Perform cross-validation to check the performance of this decoder.
        
        This function holds out data, trains new decoders using only the training data
        to check the actual performance of the current decoder.

        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        valid = ~np.isnan(self.data).any(1)
        data = self.data[valid]
        actual = self.actual[valid]

        nd = self.data.shape[1]
        dim = tuple(range(nd)), tuple(range(nd, 2*nd))

        order = np.random.permutation(len(self.data))
        idx = set(order)
        bedge = len(order) / float(blocks)

        ccs = np.zeros((blocks,))
        for b in range(blocks):
            val = order[int(b*bedge):int((b+1)*bedge)]
            trn = np.array(list(idx - set(val)))

            cal = self.__class__(data[trn], actual[trn], **self.kwargs)
            corr = np.corrcoef(cal(data[val]).T, actual[val].T)
            ccs[b] = corr[dim].mean()

        return ccs

    def __call__(self, data):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        raise NotImplementedError

class EyeProfile(Profile):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, data, actual, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(EyeProfile, self).__init__(data, actual, system="eyetracker", **kwargs)
    
    def _init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(EyeProfile, self)._init()
        valid = -(self.data == (-32768, -32768)).all(1)
        self.data = self.data[valid,:]
        self.actual = self.actual[valid,:]

class ThinPlate(Profile):
    '''Interpolates arbitrary input dimensions into arbitrary output dimensions using thin plate splines'''
    def __init__(self, data, actual, smooth=0, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.smooth = smooth
        super(ThinPlate, self).__init__(data, actual, **kwargs)
    
    def _init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(ThinPlate, self)._init()
        self.funcs = []
        from scipy.interpolate import Rbf
        for a in self.actual.T:
            f = Rbf(*np.vstack([self.data.T, a]), function='thin_plate', smooth=self.smooth)
            self.funcs.append(f)
        
    def __call__(self, data):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        raw = np.atleast_2d(data).T
        return np.array([func(*raw) for func in self.funcs]).T
    
    def __getstate__(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        state = self.__dict__.copy()
        del state['funcs']
        return state

    def __setstate__(self, state):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(ThinPlate, self).__setstate__(state)
        self._init()

class ThinPlateEye(ThinPlate, EyeProfile):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    pass

def crossval(cls, data, actual, proportion=0.7, parameter="smooth", xval_range=np.linspace(0,10,20)**2):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    actual = np.array(actual)
    data = np.array(data)

    ccs = np.zeros(len(xval_range))
    for i, smooth in enumerate(xval_range):
        cal = cls(data, actual, **{parameter:smooth})
        ccs[i] = cal.performance().mean()
    
    best = xval_range[ccs.argmax()]
    return cls(data, actual, **{parameter:best}), best, ccs

class Affine(Profile):
    '''Runs a linear affine interpolation between data and actual'''
    def __init__(self, data, actual):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.data = data
        self.actual = actual
        #self.xfm = np.linalg.lstsq()




class AutoAlign(object):
    '''Runs the autoalignment filter to center everything into the chair coordinates'''
    def __init__(self, reference):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        print("Making autoaligner from reference %s"%reference)
        from riglib.stereo_opengl import xfm
        self._quat = xfm.Quaternion
        self.ref = np.load(reference)['reference']
        self.xfm = xfm.Quaternion()
        self.off1 = np.array([0,0,0])
        self.off2 = np.array([0,0,0])

    def __call__(self, data):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        mdata = data.mean(0)[:, :3]
        avail = (data[:,-6:, -1] > 0).all(0)
        if avail[:3].all():
            #ABC reference
            cdata = mdata[-6:-3] - mdata[-6]
            self.off1 = mdata[-6]
            self.off2 = self.ref[0]
            rot1 = self._quat.rotate_vecs(cdata[1], self.ref[1] - self.ref[0])
            rot2 = self._quat.rotate_vecs((rot1*cdata[2]), self.ref[2] - self.ref[0])
            self.xfm = rot2*rot1
        elif avail[3:].all():
            #DEF reference
            cdata = mdata[-3:] - mdata[-3]
            self.off1 = mdata[-3]
            self.off2 = self.ref[3]
            rot1 = self._quat.rotate_vecs(cdata[1], self.ref[4] - self.ref[3])
            rot2 = self._quat.rotate_vecs((rot1*cdata[2]), self.ref[5] - self.ref[3])
            self.xfm = rot2*rot1
        rdata = self.xfm*(mdata[:-6] - self.off1) + self.off2
        rdata[(data[:,:-6,-1] < 0).any(0)] = np.nan
        return np.hstack([rdata, np.ones((len(rdata),1))])[np.newaxis]
