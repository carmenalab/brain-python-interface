'''This module implements IK functions for running inverse kinematics.
Current, only a two-joint system can be modelled. I am not skilled in the ways of robotics...'''
from __future__ import division
import numpy as np

from xfm import Quaternion

class TwoJoint(object):
    '''Models a two-joint IK system (arm, leg, etc). Constrains the system by 
    always having middle joint halfway between the origin and target'''
    def __init__(self, origin_bone, target_bone, lengths=(20,20)):
        '''Takes two Model objects for the "upperarm" bone and the "lowerarm" bone.
        Assumes the models start at origin, with vector to (0,0,1) for bones'''
        self.upperarm = origin_bone
        self.forearm = target_bone
        self.lengths = lengths
        self.tlen = lengths[0] + lengths[1]

    def _midpos(self, target):
        m, n = self.lengths
        x, y, z = target

        #this heinous equation brought to you by Wolfram Alpha
        #it is ONE of the solutions to this system of equations:
        # a^2 + b^2 + (z/2)^2 = m^2
        # (x-a)^2 + (y-b)^2 + (z/2)^2 = n^2
        a = (m**2*x**2+y*np.sqrt(-x**2*(m**4-2*m**2*n**2-2*m**2*x**2-2*m**2*y**2+n**4-2*n**2*x**2-2*n**2*y**2+x**4+2*x**2*y**2+x**2*z**2+y**4+y**2*z**2))-n**2*x**2+x**4+x**2*y**2)/(2*x*(x**2+y**2))
        b = (m**2*y-np.sqrt(-x**2*(m**4-2*m**2*n**2-2*m**2*x**2-2*m**2*y**2+n**4-2*n**2*x**2-2*n**2*y**2+x**4+2*x**2*y**2+x**2*z**2+y**4+y**2*z**2))-n**2*y+x**2*y+y**3)/(2*(x**2+y**2))

        return a, b, z/2

    def set(self, target):
        assert np.linalg.norm(target) <= self.tlen
        elbow = np.array(self._midpos(target))
        
        self.upperarm.xfm.rotate = Quaternion.rotate_vecs((0,0,1), elbow)
        self.upperarm._recache_xfm()
        self.lowerarm.xfm.rotate = Quaternion.rotate_vecs((0,0,1), np.array(target)-elbow)
        self.lowerarm._xfm = self.lowerarm.xfm.to_mat()