'''This module implements IK functions for running inverse kinematics.
Current, only a two-joint system can be modelled. I am not skilled in the ways of robotics...'''
from __future__ import division
import numpy as np

from xfm import Quaternion
from models import Group
from primitives import Cylinder, Sphere
from textures import TexModel
from utils import cloudy_tex

class RobotArm2D(Group):
    '''
    A basic robot arm plant. One link with one rotating joint. Joints have one degree of freedom.
    Initialized with anchor point at origin and joint moving in x-z plane. Takes coordinates in 3D
    for future compatibility with tasks but all movements restricted to x-z plane and y coordinates ignored.
    '''
    def __init__(self, link_radii=[.2], joint_radii=[.5],link_lengths=[5], joint_colors = [(1,1,1,1)],
        link_colors = [(1,1,1,1)], **kwargs):
        self.link_radii = link_radii
        self.joint_radii = joint_radii
        self.link_lengths = link_lengths
        self.joint_colors = joint_colors
        self.link_colors = link_colors
        self.curr_vecs = np.zeros([1,3])
        self.curr_vecs[0,:] = np.array([self.link_lengths[0],0,0])
        self.link1 = Group((Cylinder(radius=link_radii[0], height=link_lengths[0], color=link_colors[0]), Sphere(radius=joint_radii[0],color=joint_colors[0])))
        super(RobotArm2D, self).__init__([self.link1], **kwargs)

    def get_endpoint_pos(self):
        '''
        Returns the current position of the non-anchored end of the arm.
        '''
        return self.curr_vecs[0,:]

    def set_endpoint_pos(self,x,y,z):
        '''
        Positions the arm according to specified endpoint position.
        '''
        
        #normalize vector from origin to endpoint
        mag = np.linalg.norm([x,0,z]) #project onto xz plane
        normed = np.array([x,0,z])/mag

        #set link vector to aligned with endpoint. if endpoint is not on circle, endpoint will
        #end up at closest possible point on circle
        self.curr_vecs[0,:] = normed*self.link_lengths[0]
        self._update_links()

    def get_joint_pos(self):
        '''
        Returns the joint angles of the arm in radians
        '''
        return np.arctan2(self.curr_vecs[0,2],self.curr_vecs[0,0])
        # if self.curr_vecs[0,2]>=0:
        #     return np.arcsin(self.curr_vecs[0,2]/self.link_lengths[0])
        # else:
        #     print "here"
        #     return -1*np.arcsin(self.curr_vecs[0,2]/self.link_lengths[0])
        

    def set_joint_pos(self,theta):
        '''
        Set the joint by specifying the angle in radians.
        '''
        if theta is not None:
            xs = self.link_lengths[0]*np.cos(theta)
            ys = 0.0
            zs = self.link_lengths[0]*np.sin(theta)
            self.curr_vecs[0,:] = np.array([xs, ys, zs])
        self._update_links()   

    def _update_links(self):
        self.link1.xfm.rotate = Quaternion.rotate_vecs((0,0,1), self.curr_vecs[0,:]).norm()
        self.link1._recache_xfm()


class RobotArm2J2D(RobotArm2D):
    '''
    A 2 joint version of the 2D robot arm plant.
    '''

    def __init__(self, link_radii=[.2, .2], joint_radii=[.5, .5],link_lengths=[5, 5], joint_colors = [(1,1,1,1), (1,1,1,1)],
        link_colors = [(1,1,1,1), (1,1,1,1)], **kwargs):
        self.link_radii = link_radii
        self.joint_radii = joint_radii
        self.link_lengths = link_lengths
        self.joint_colors = joint_colors
        self.link_colors = link_colors
        self.curr_vecs = np.zeros([2,3])
        self.curr_vecs[0,:] = np.array([self.link_lengths[0],0,0])
        self.curr_vecs[1,:] = np.array([self.link_lengths[1],0,0]) #curr_vecs stores the current relative vectors of the arm links
        
        self.link2 = Group((Cylinder(radius=link_radii[1], height=link_lengths[1], color=link_colors[1]), Sphere(radius=joint_radii[1],color=joint_colors[1])))
        self.link1 = Group((Cylinder(radius=link_radii[0], height=link_lengths[0], color=link_colors[0]), Sphere(radius=joint_radii[0],color=joint_colors[0]))).translate(0,0,self.link_lengths[1])
        self.link_group_1 = Group([self.link2, self.link1])
        super(RobotArm2D, self).__init__([self.link_group_1], **kwargs)

    def _update_links(self):
        self.link_group_1.xfm.rotate = Quaternion.rotate_vecs((0,0,1), self.curr_vecs[1,:]).norm()
        self.link_group_1._recache_xfm()
        super(RobotArm2J2D, self)._update_links()

    def set_endpoint_pos(self,x,y,z):
        '''
        Positions the arm according to specified endpoint position. Uses 2D inverse kinematic equations to calculate joint angles.
        '''
        pass

    def get_joint_pos(self):
        '''
        Returns the joint angles of the arm in radians
        '''
        link2ang = np.arctan2(self.curr_vecs[1,2], self.curr_vecs[1,0])
        link1ang = super(RobotArm2J2D, self).get_joint_pos()
        return np.array([link1ang, link2ang])
        

    def set_joint_pos(self,theta):
        '''
        Set the joint by specifying the angle in radians. Theta is a list of angles. If an element of theta = None, angle should remain the same.
        '''
        if theta[1] is not None:
            self.curr_vecs[1,:] = np.array([self.link_lengths[1]*np.cos(theta[1]), 0.0, self.link_lengths[1]*np.sin(theta[1])])
        if theta[0] is not None: theta[0] = theta[0] + np.pi/2 # add pi/2 to lower link angle b/c absolute 0 is horizontal but we want 0 to be aligned with upper link which rotates from 0,0,1 vector
        super(RobotArm2J2D, self).set_joint_pos(theta[0])  


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
        self.curr_angles = np.zeros(2)

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

        elbow_angle_mod = elbow_angle + np.pi/2

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
        xe = self.lengths[1]*np.cos(elbow_angle_mod)
        ye = 0.0
        ze = self.lengths[1]*np.sin(elbow_angle_mod)
        # Find absolute vector
        xe2 = self.lengths[1]*np.cos(shoulder_angle+elbow_angle)
        ye2 = 0.0
        ze2 = self.lengths[1]*np.sin(shoulder_angle+elbow_angle)
        self.curr_vecs[1,:] = np.array([xe2, ye2, ze2])
        self.forearm.xfm.rotate = Quaternion.rotate_vecs((0,0,1), (xe,0,ze)).norm()

        self.curr_angles[0] = shoulder_angle
        self.curr_angles[1] = elbow_angle_mod

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
    def __init__(self, link_radii=(.2, .2), ball_radii=(.5,.5),lengths=(5, 4), ball_colors = ((1,1,1,1),(1,1,1,1)),\
        link_colors = ((1,1,1,1), (1,1,1,1)), **kwargs):
        self.link_radii = link_radii
        self.ball_radii = ball_radii
        self.lengths = lengths
        self.forearm = Group([
            Cylinder(radius=link_radii[1], height=lengths[1], color=link_colors[1]), 
            Sphere(radius=ball_radii[1],color=ball_colors[1]).translate(0, 0, lengths[1])]).translate(0,0,lengths[0])
        self.upperarm = Group([
            Cylinder(radius=link_radii[0], height=lengths[0],color=link_colors[0]), 
            Sphere(radius=ball_radii[0],color=ball_colors[0]).translate(0, 0, lengths[0]),
            self.forearm])
        self.system = TwoJoint(self.upperarm, self.forearm, lengths = (self.lengths))
        super(RobotArm, self).__init__([self.upperarm], **kwargs)

    def set_endpoint_2D(self, target):
        self.system.set_endpoint_2D(target)

    def set_joints_2D(self, shoulder_angle, elbow_angle): 
        self.system.set_joints_2D(shoulder_angle, elbow_angle)

    def get_hand_location(self, shoulder_anchor):
        ''' returns position of ball at end of forearm (hand)'''
        return shoulder_anchor + self.system.curr_vecs[0] +self.system.curr_vecs[1]

    def get_joint_angles_2D(self):
        return self.system.curr_angles[0], self.system.curr_angles[1] - np.pi/2