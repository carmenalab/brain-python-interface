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
        self.curr_vecs = np.zeros([2,3])

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

    def set_endpoint_3D(self, target):
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

    def set_endpoint_2D(self, target):
        ''' Given an endpoint coordinate in the x-z plane, solves for joint positions in that plane via inverse kinematics'''
        pass

    def set_joints_2D(self, shoulder_angle, elbow_angle):
        ''' Given angles for shoulder and elbow in a plane, set joint positions. Shoulder angle is in fixed
        frame of reference where 0 is horizontal pointing to the right (left if viewing on screen without mirror), pi/2
        is vertical pointing up, pi is horizontal pointing to the left. Elbow angle is relative to upper arm vector, where
        0 is fully extended, pi/2 is a right angle to upper arm pointing left, and pi is fully overlapping with upper
        arm.'''

        elbow_angle = elbow_angle + np.pi/2

        #if shoulder_angle>np.pi: shoulder_angle = np.pi
        #if shoulder_angle<0.0: shoulder_angle = 0.0
        #if elbow_angle>np.pi: elbow_angle = np.pi
        #if elbow_angle<0: elbow_angle = 0

        # Find upper arm vector
        xs = self.lengths[0]*np.cos(shoulder_angle)
        ys = 0.0
        zs = self.lengths[0]*np.sin(shoulder_angle)
        self.curr_vecs[0,:] = np.array([xs, ys, zs])
        self.upperarm.xfm.rotate = Quaternion.rotate_vecs((0,0,1), (xs,0,zs)).norm()

        # Find forearm vector (relative to upper arm)
        xe = self.lengths[1]*np.cos(elbow_angle)
        ye = 0.0
        ze = self.lengths[1]*np.sin(elbow_angle)
        # Find absolute vector
        xe2 = self.lengths[1]*np.cos(shoulder_angle+elbow_angle)
        ye2 = 0.0
        ze2 = self.lengths[1]*np.sin(shoulder_angle+elbow_angle)
        self.curr_vecs[1,:] = np.array([xe, ye, ze])
        self.forearm.xfm.rotate = Quaternion.rotate_vecs((0,0,1), (xe,0,ze)).norm()

        self.upperarm._recache_xfm()


## Original, written by james

# TexCylinder = type("TexCylinder", (Cylinder, TexModel), dict())
# class RobotArm(Group):
#     def __init__(self, radii=(2, 1.5), lengths=(15, 20), **kwargs):
#         tex = cloudy_tex()
#         self.forearm = Group([
#             TexCylinder(radius=radii[1], height=lengths[1], tex=tex, shininess=50), 
#             Sphere(radii[1]+0.5).translate(0, 0, lengths[1])]).translate(0,0,lengths[0])
#         self.upperarm = Group([
#             Sphere(radii[0]+0.5),
#             TexCylinder(radius=radii[0], height=lengths[0], tex=tex, shininess=50), 
#             Sphere(radii[0]+0.5).translate(0, 0, lengths[0]),
#             self.forearm])
#         self.system = TwoJoint(self.upperarm, self.forearm)
#         super(RobotArm, self).__init__([self.upperarm], **kwargs)

#     def set_endpoint_2D(self, target):
#         self.system.set_endpoint_2D(target)

#     def set_joints_2D(self, shoulder_angle, elbow_angle):
#         ''' returns position of ball at end of forearm (hand)'''
#         self.system.set_joints_2D(shoulder_angle, elbow_angle)
#         return self.forearm.models[1].xfm.move


## added by helene to prep for arm task

class RobotArm(Group):
    def __init__(self, link_radii=(.2, .2), ball_radii=(.5,.5),lengths=(5, 4), **kwargs):
        self.link_radii = link_radii
        self.ball_radii = ball_radii
        self.lengths = lengths
        self.forearm = Group([
            Cylinder(radius=link_radii[1], height=lengths[1], color=(0,0,.5,1)), 
            Sphere(radius=ball_radii[1],color=(1,1,1,.2)).translate(0, 0, lengths[1])]).translate(0,0,lengths[0])
        self.upperarm = Group([
            Cylinder(radius=link_radii[0], height=lengths[0],color=(0,0,1,1)), 
            Sphere(radius=ball_radii[0],color=(1,1,1,1)).translate(0, 0, lengths[0]),
            self.forearm])
        self.system = TwoJoint(self.upperarm, self.forearm, lengths = (self.lengths))
        super(RobotArm, self).__init__([self.upperarm], **kwargs)

    def set_endpoint_2D(self, target):
        self.system.set_endpoint_2D(target)

    def set_joints_2D(self, shoulder_angle, elbow_angle): 
        self.system.set_joints_2D(shoulder_angle, elbow_angle)

    def get_hand_location(self, shoulder_anchor):
        ''' returns position of ball at end of forearm (hand)'''
        upper_vec = self.system.curr_vecs[0]
        # adjust lower arm vector to include radius of ball at the end
        lower_vec = (self.system.curr_vecs[1]/self.lengths[1])*(self.lengths[1]+self.ball_radii[1])
        return shoulder_anchor + upper_vec + lower_vec