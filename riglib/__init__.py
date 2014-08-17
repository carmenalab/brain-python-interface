'''
riglib module imports
'''
import warnings

from . import experiment
from . import blackrock
from . import bmi
from . import brainamp
from . import calibrations
from . import eyetracker
from . import hdfwriter
from . import ismore
from . import motiontracker
from . import mp_calc
from . import nidaq
try:
    from . import phidgets
except ImportError:
    warnings.warn('Phidgets import problem!')

from . import plexon
from . import reward
from . import stereo_opengl
# from . import loc_config


class FuncProxy(object):
    '''
    This class is similar to tasktrack.FuncProxy. Used by source.py
    '''
    def __init__(self, name, pipe, event):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.pipe = pipe
        self.name = name
        self.event = event

    def __call__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.pipe.send((self.name, args, kwargs))
        self.event.set()
        return self.pipe.recv()

from . import sink
from . import source
