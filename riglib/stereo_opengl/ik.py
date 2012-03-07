'''This module implements IK functions for running inverse kinematics.
Current, only a two-joint system can be modelled. I am not skilled in the ways of robotics...'''
from __future__ import division
import numpy as np

from xfm import Quaternion
from models import Group
from primitives import Cylinder, Sphere
from textures import TexModel
from utils import cloudy_tex

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
        if x > 0:
            a = (m**2*x**2+y*np.sqrt(-x**2*(m**4-2*m**2*n**2-2*m**2*x**2-2*m**2*y**2+n**4-2*n**2*x**2-2*n**2*y**2+x**4+2*x**2*y**2+x**2*z**2+y**4+y**2*z**2))-n**2*x**2+x**4+x**2*y**2)/(2*x*(x**2+y**2))
            b = (m**2*y-np.sqrt(-x**2*(m**4-2*m**2*n**2-2*m**2*x**2-2*m**2*y**2+n**4-2*n**2*x**2-2*n**2*y**2+x**4+2*x**2*y**2+x**2*z**2+y**4+y**2*z**2))-n**2*y+x**2*y+y**3)/(2*(x**2+y**2))
        else:
            a = (m**2*x**2-y*np.sqrt(-x**2*(m**4-2*m**2*n**2-2*m**2*x**2-2*m**2*y**2+n**4-2*n**2*x**2-2*n**2*y**2+x**4+2*x**2*y**2+x**2*z**2+y**4+y**2*z**2))-n**2*x**2+x**4+x**2*y**2)/(2*x*(x**2+y**2))
            b = (m**2*y+np.sqrt(-x**2*(m**4-2*m**2*n**2-2*m**2*x**2-2*m**2*y**2+n**4-2*n**2*x**2-2*n**2*y**2+x**4+2*x**2*y**2+x**2*z**2+y**4+y**2*z**2))-n**2*y+x**2*y+y**3)/(2*(x**2+y**2))
        return a, b, z/2

    def set(self, target):
        '''Sets the endpoint coordinate for the two-joint system'''
        #Make sure the target is actually achievable
        if np.linalg.norm(target) > self.tlen:
            self.upperarm.xfm.rotate = Quaternion.rotate_vecs((0,0,1), target).norm()
            self.forearm.xfm.rotate = Quaternion()
        else:
            elbow = np.array(self._midpos(target))
            
            #rotate the upperarm to the elbow
            self.upperarm.xfm.rotate = Quaternion.rotate_vecs((0,0,1), elbow).norm()

            #this broke my mind for 2 hours at least, so I cheated
            #Rotate first to (0,0,1), then rotate to the target-elbow
            self.forearm.xfm.rotate = (Quaternion.rotate_vecs(elbow, (0,0,1)) *
                Quaternion.rotate_vecs((0,0,1), target-elbow)).norm()
        
        self.upperarm._recache_xfm()

TexCylinder = type("TexCylinder", (Cylinder, TexModel), dict())
class RobotArm(Group):
    def __init__(self, radii=(2, 1.5), lengths=(15, 20), **kwargs):
        tex = cloudy_tex()
        self.forearm = Group([
            TexCylinder(radius=radii[1], height=lengths[1], tex=tex, shininess=50), 
            Sphere(radii[1]+0.5).translate(0, 0, lengths[1])]).translate(0,0,lengths[0])
        self.upperarm = Group([
            Sphere(radii[0]+0.5),
            TexCylinder(radius=radii[0], height=lengths[0], tex=tex, shininess=50), 
            Sphere(radii[0]+0.5).translate(0, 0, lengths[0]),
            self.forearm])
        self.system = TwoJoint(self.upperarm, self.forearm)
        super(RobotArm, self).__init__([self.upperarm], **kwargs)

    def set(self, target):
        self.system.set(target)
