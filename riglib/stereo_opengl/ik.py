'''
This module implements IK functions for running inverse kinematics.
Current, only a two-joint system can be modelled.
'''
from __future__ import division
import numpy as np

from xfm import Quaternion
from models import Group
from primitives import Cylinder, Sphere, Cone
from textures import TexModel
from utils import cloudy_tex
from collections import OrderedDict

joint_angles_dtype = [('sh_pflex', np.float64), ('sh_pabd', np.float64), ('sh_prot', np.float64), ('el_pflex', np.float64), ('el_psup', np.float64)]
joint_vel_dtype = [('sh_vflex', np.float64), ('sh_vabd', np.float64), ('sh_vrot', np.float64), ('el_vflex', np.float64), ('el_vsup', np.float64)]
def get_arm_class_list():
    return [RobotArm2D, RobotArm2J2D]

def inv_kin_2D(pos, l_upperarm, l_forearm, vel=None):
    '''
    Inverse kinematics for a 2D arm. This function returns all 5 angles required
    to specify the pose of the exoskeleton (see riglib.bmi.train for the 
    definitions of these angles). This pose is constrained to the x-z plane
    by forcing shoulder flexion/extension, elbow rotation and supination/pronation
    to always be 0. 
    '''
    if np.ndim(pos) == 1:
        pos = pos.reshape(1,-1)

    # require the y-coordinate to be 0, i.e. flat on the screen
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    assert np.all(y == 0)

    if vel is not None:
        if np.ndim(vel) == 1:
            vel = vel.reshape(1,-1)
        assert pos.shape == vel.shape
        vx, vy, vz = vel[:,0], vel[:,1], vel[:,2]
        assert np.all(vy == 0)

    L = np.sqrt(x**2 + z**2)
    cos_el_pflex = (L**2 - l_forearm**2 - l_upperarm**2) / (2*l_forearm*l_upperarm)

    cos_el_pflex[ (cos_el_pflex > 1) & (cos_el_pflex < 1 + 1e-9)] = 1
    el_pflex = np.arccos(cos_el_pflex)

    sh_pabd = np.arctan2(z, x) - np.arcsin(l_forearm * np.sin(np.pi - el_pflex) / L)
    angles = np.zeros(len(pos), dtype=joint_angles_dtype)
    angles['sh_pabd'] = sh_pabd
    angles['el_pflex'] = el_pflex
    if np.any(np.isnan(angles['el_pflex'])) or np.any(np.isnan(angles['sh_pabd'])):
        print "position = ", pos
        print "angles = ", angles['el_pflex'], angles['sh_pabd']
        print "L = ", L
        print "cos_el_pflex = ", cos_el_pflex
        print "np.arctan2(z, x = ", (np.arctan2(z, x))
        print "np.arcsin(l_forearm * np.sin(np.pi - el_pflex) = ", (np.arcsin(l_forearm * np.sin(np.pi - el_pflex)))

    if vel is not None:
        joint_vel = np.zeros(len(pos), dtype=joint_vel_dtype)
        # if len(vel) > 0:
        #     raise NotImplementedError
        
        # Calculate the jacobian
        for k, angle in enumerate(angles):
            s1 = np.sin(angle['sh_pabd'])
            s2 = np.sin(angle['sh_pabd'] + angle['el_pflex'])
            c1 = np.cos(angle['sh_pabd'])
            c2 = np.cos(angle['sh_pabd'] + angle['el_pflex'])           
            J = np.array([[-l_upperarm*s1-l_forearm*s2, -l_forearm*s2 ],
                          [ l_upperarm*c1+l_forearm*c2,  l_forearm*c2 ]])
            J_inv = np.linalg.inv(J)

            joint_vel_mat = np.dot(J_inv, vel[k, [0,2]])
            joint_vel[k]['sh_vabd'], joint_vel[k]['el_vflex'] = joint_vel_mat.ravel()

        return angles, joint_vel
    else:
        return angles    

class RobotArm2D(Group):
    '''
    A basic robot arm plant. One link with one rotating joint. Joints have one degree of freedom.
    Initialized with anchor point at origin and joint moving in x-z plane. Takes coordinates in 3D
    for future compatibility with tasks but all movements restricted to x-z plane and y coordinates ignored.
    '''
    def __init__(self, link_radii=[.2], joint_radii=[.5],link_lengths=[5], joint_colors = [(1,1,1,1)],
        link_colors = [(1,1,1,1)], **kwargs):
        self.num_joints = 1
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

        #set link vector to be aligned with endpoint. if endpoint is not on circle, endpoint will
        #end up at closest possible point on circle
        self.curr_vecs[0,:] = normed*self.link_lengths[0]
        self._update_links()

    def get_joint_pos(self):
        '''
        Returns the joint angles of the arm in radians
        '''
        return np.arctan2(self.curr_vecs[0,2],self.curr_vecs[0,0])
        

    def set_joint_pos(self,theta):
        '''
        Set the joint by specifying the angle in radians.
        '''
        if theta is not None and ~np.isnan(theta):
            xs = self.link_lengths[0]*np.cos(theta)
            ys = 0.0
            zs = self.link_lengths[0]*np.sin(theta)
            self.curr_vecs[0,:] = np.array([xs, ys, zs])
        self._update_links()   

    def _update_links(self):
        self.link1.xfm.rotate = Quaternion.rotate_vecs((1,0,0), self.curr_vecs[0]).norm()
        self.link1._recache_xfm()

class CursorPlant(object):
    def __init__(self, **kwargs):
        self.position = np.zeros(3)

    def get_endpoint_pos(self):
        return self.position

    def set_endpoint_pos(self,pt):
        self.position = pt

    def get_joint_pos(self):
        raise ValueError("cursor has no joints!")

    def set_joint_pos(self,theta):
        raise ValueError("Cursor has no joints!")

class RobotArm2J2D(RobotArm2D):
    '''
    A 2 joint version of the 2D robot arm plant.
    '''

    def __init__(self, link_radii=[.2, .2], joint_radii=[.5, .5],link_lengths=[5, 5], joint_colors = [(1,1,1,1), (1,1,1,1)],
        link_colors = [(1,1,1,1), (1,1,1,1)], **kwargs):
        self.num_joints = 2
        self.link_radii = link_radii
        self.joint_radii = joint_radii
        self.link_lengths = link_lengths
        self.joint_colors = joint_colors
        self.link_colors = link_colors
        self.curr_vecs = np.zeros([2,3])

        self.curr_vecs[:,0] = self.link_lengths
        
        self.link2 = Group((Cylinder(radius=link_radii[1], height=link_lengths[1], color=link_colors[1]), Sphere(radius=joint_radii[1],color=joint_colors[1])))
        self.link1 = Group((Cone(radius1=link_radii[0], radius2 = link_radii[1]/2, height=link_lengths[0], color=link_colors[0]), Sphere(radius=joint_radii[0],color=joint_colors[0]))).translate(0,0,self.link_lengths[1])
        self.link_group_1 = Group([self.link2, self.link1])

        super(RobotArm2D, self).__init__([self.link_group_1], **kwargs)

    def _update_links(self):
        arg1 = (0,0,1)
        arg2 = self.curr_vecs[1,:]
        self.link_group_1.xfm.rotate = Quaternion.rotate_vecs(arg1,arg2).norm()
        self.link_group_1._recache_xfm()
        super(RobotArm2J2D, self)._update_links()

    def get_endpoint_pos(self):
        '''
        Returns the current position of the non-anchored end of the arm.
        '''
        relangs = np.arctan2(self.curr_vecs[:,2], self.curr_vecs[:,0])
        return self.perform_fk(relangs)      

    def perform_fk(self, angs):
        abselang = np.sum(angs)
        abselvec = self.link_lengths[0]*np.array([np.cos(abselang), 0, np.sin(abselang)])
        shvec = self.link_lengths[1]*np.array([np.cos(angs[1]), 0, np.sin(angs[1])])
        return abselvec + shvec

    def set_endpoint_pos(self,pos):
        '''
        Positions the arm according to specified endpoint position. Uses 2D inverse kinematic equations to calculate joint angles.
        '''
        if pos is not None:
            angles = self.perform_ik(pos)
            self.set_joint_pos([angles['el_pflex'], angles['sh_pabd']])

    def perform_ik(self,pos):
        x,y,z = pos
        # set y to 0 for 2D
        y=0
        # if position is out of reach, set it to nearest point that can be reached
        mag = np.linalg.norm([x,y,z])
        normed = np.array([x,y,z])/mag
        if mag>np.sum(self.link_lengths):    
            x,y,z = normed*np.sum(self.link_lengths)
        if mag<np.abs(np.diff(self.link_lengths)):
            x,y,z = normed*np.abs(np.diff(self.link_lengths))

        return inv_kin_2D(np.array([x,y,z]), self.link_lengths[1], self.link_lengths[0])

    def calc_joint_angles(self, vecs):
        angs = np.arctan2(vecs[:,2], vecs[:,0])
        return angs

    def get_joint_pos(self):
        '''
        Returns the joint angles of the arm in radians
        '''
        return self.calc_joint_angles(self.curr_vecs)
        
    def set_joint_pos(self,theta):
        '''
        Set the joint by specifying the angle in radians. Theta is a list of angles. If an element of theta = NaN, angle should remain the same.
        '''
        if theta[1] is not None and ~np.isnan(theta[1]):
            self.curr_vecs[1,:] = np.array([self.link_lengths[1]*np.cos(theta[1]), 0.0, self.link_lengths[1]*np.sin(theta[1])])
        super(RobotArm2J2D, self).set_joint_pos(theta[0])


class RobotArm5J2D(RobotArm2J2D):

    def __init__(self, link_radii=[.2,.2,.2,.2,.2], joint_radii=[.5,.5,.5,.5,.5],link_lengths=[2,2,2,2,2], joint_colors = [(1,1,1,1), (1,1,1,1), (1,1,1,1),(1,1,1,1),(1,1,1,1)],
        link_colors = [(1,1,1,1), (1,1,1,1),(1,1,1,1),(1,1,1,1),(1,1,1,1)], **kwargs):
        self.num_joints = 5
        self.link_radii = link_radii
        self.joint_radii = joint_radii
        self.link_lengths = link_lengths
        self.joint_colors = joint_colors
        self.link_colors = link_colors
        self.curr_vecs = np.zeros([5,3]) #row 0 is most distal link

        self.curr_vecs[:,0] = self.link_lengths

        self.joint_planes = [1,1,1,1,0] #indicates which axis each joint does NOT move in (x=0,y=1,z=2), distal to proximal
        
        self.link5 = Group((Cylinder(radius=link_radii[4], height=link_lengths[4], color=link_colors[4]), Sphere(radius=joint_radii[4],color=joint_colors[4])))
        self.link4 = Group((Cylinder(radius=link_radii[3], height=link_lengths[3], color=link_colors[3]), Sphere(radius=joint_radii[3],color=joint_colors[3])))
        self.link3 = Group((Cylinder(radius=link_radii[2], height=link_lengths[2], color=link_colors[2]), Sphere(radius=joint_radii[2],color=joint_colors[2])))
        self.link2 = Group((Cylinder(radius=link_radii[1], height=link_lengths[1], color=link_colors[1]), Sphere(radius=joint_radii[1],color=joint_colors[1])))
        self.link1 = Group((Cone(radius1=link_radii[0], radius2 = link_radii[1]/2, height=link_lengths[0], color=link_colors[0]), Sphere(radius=joint_radii[0],color=joint_colors[0]))).translate(0,0,self.link_lengths[1])
        
        self.link_group_1 = Group([self.link2, self.link1]).translate(0,0,self.link_lengths[2])
        self.link_group_2 = Group([self.link3, self.link_group_1]).translate(0,0,self.link_lengths[3])
        self.link_group_3 = Group([self.link4, self.link_group_2]).translate(0,0,self.link_lengths[4])
        self.link_group_4 = Group([self.link5, self.link_group_3])

        super(RobotArm2D, self).__init__([self.link_group_4], **kwargs)

    def _update_links(self):
        self.link_group_4.xfm.rotate = Quaternion.rotate_vecs((0,0,1),self.curr_vecs[4,:]).norm()
        self.link_group_4._recache_xfm()
        self.link_group_3.xfm.rotate = Quaternion.rotate_vecs((0,0,1),self.curr_vecs[3,:]).norm()
        self.link_group_3._recache_xfm()
        self.link_group_2.xfm.rotate = Quaternion.rotate_vecs((0,0,1),self.curr_vecs[2,:]).norm()
        self.link_group_2._recache_xfm()
        super(RobotArm5J2D, self)._update_links()

    def get_endpoint_pos(self):
        '''
        Returns the current position of the non-anchored end of the arm.
        '''
        relangs = np.zeros(5)
        for i in range(5):
            ref = np.array([0,1,2])
            axes = ref[ref!=self.joint_planes[i]]
            relangs[i] = np.arctan2(self.curr_vecs[i,axes[1]], self.curr_vecs[i,axes[0]])
        return self.perform_fk(relangs)      

    def perform_fk(self, angs):
        absvecs = np.zeros([5,3])
        for i in range(5):
            if i>0:
                angs_t = angs[:-i]
                jp_t = self.joint_planes[:-i]
            else:
                angs_t = angs
                jp_t = self.joint_planes
            abs_yz_ang = np.sum(angs[jp_t[jp_t==0]])
            abs_xz_ang = np.sum(angs[jp_t[jp_t==1]])
            abs_xy_ang = np.sum(angs[jp_t[jp_t==2]])
            absvecs[i] = self.link_lengths[i]*np.array([np.cos(abs_xz_ang), np.sin(abs_xy_ang), np.sin(abs_xz_ang)])
        return np.sum(absvecs,axis=0)

    def set_endpoint_pos(self,pos):
        # '''
        # Positions the arm according to specified endpoint position. Uses 2D inverse kinematic equations to calculate joint angles.
        # '''
        # if pos is not None:
        #     angles = self.perform_ik(pos)
        #     self.set_joint_pos([angles['el_pflex'], angles['sh_pabd']])
        pass

    def perform_ik(self,pos):
        x,y,z = pos
        # set y to 0 for 2D
        y=0
        # if position is out of reach, set it to nearest point that can be reached
        mag = np.linalg.norm([x,y,z])
        normed = np.array([x,y,z])/mag
        if mag>np.sum(self.link_lengths):    
            x,y,z = normed*np.sum(self.link_lengths)
        if mag<np.abs(np.diff(self.link_lengths)):
            x,y,z = normed*np.abs(np.diff(self.link_lengths))

        return inv_kin_2D(np.array([x,y,z]), self.link_lengths[1], self.link_lengths[0])

    def calc_joint_angles(self, vecs):
        angs = np.zeros([5])
        for i in range(5):
            ref = np.array([0,1,2])
            axes = ref[ref!=self.joint_planes[i]]
            angs[i] = np.arctan2(vecs[i,axes[1]], vecs[i,axes[0]])
        return angs

    def get_joint_pos(self):
        '''
        Returns the joint angles of the arm in radians
        '''
        return self.calc_joint_angles(self.curr_vecs)
        
    def set_joint_pos(self,theta):
        '''
        Set the joint by specifying the angle in radians. Theta is a list of angles. If an element of theta = NaN, angle should remain the same.
        '''
        for i in range(len(theta)):
            if theta[i] is not None and ~np.isnan(theta[i]):
                vec = np.zeros([3])
                ref = np.array([0,1,2])
                axes = ref[ref!=self.joint_planes[i]]
                vec[axes] = self.link_lengths[i]*np.array([np.cos(theta[i]), np.sin(theta[i])])
                self.curr_vecs[i] = vec

        self._update_links()

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
