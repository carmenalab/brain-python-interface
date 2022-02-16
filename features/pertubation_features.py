'''
Features to perform pertubation of cursor control
'''

import numpy as np

from riglib.experiment import traits
from scipy.spatial.transform import Rotation as R

class PertubationFeature(traits.HasTraits):
    '''
    Enable reading of data from touch sensor
    '''

    pertubation_rotation = traits.Float(0.0, desc="rotation in the x,y plane in degrees")

    def _transform_coords(self, coords):
        rot = R.from_euler('z', self.pertubation_rotation, degrees=True)
        return np.matmul(rot.as_matrix(), coords)