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
from riglib.bmi import robot_arms

pi = np.pi

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

    def set_endpoint_pos(self, x, y, z, **kwargs):
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
        return np.arctan2(self.curr_vecs[0,2], self.curr_vecs[0,0])
        

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

    def set_endpoint_pos(self, pt, **kwargs):
        self.position = pt

    def get_joint_pos(self):
        raise ValueError("cursor has no joints!")

    def set_joint_pos(self,theta):
        raise ValueError("Cursor has no joints!")

def make_list(value, num_joints):
    '''
    Helper function to allow joint/link properties of the chain to be specified
    as one value for all joints/links or as separate values for each
    '''
    if isinstance(value, list) and len(value) == num_joints:
        return value
    else:
        return [value] * num_joints

arm_color = (181/256., 116/256., 96/256., 1)
arm_radius = 0.6

class RobotArmGen2D(Group):
    def __init__(self, num_joints=4, link_radii=arm_radius, joint_radii=arm_radius, link_lengths=[15,15,5,5], joint_colors=arm_color,
        link_colors=arm_color, **kwargs):

        # For now inverse kinematics is specific to 4 joints
        assert num_joints in [2, 4], "IK only works for chains of certain length!"
        self.num_joints = num_joints

        self.link_radii = make_list(link_radii, num_joints)
        self.joint_radii = make_list(joint_radii, num_joints)
        self.link_lengths = make_list(link_lengths, num_joints)
        self.joint_colors = make_list(joint_colors, num_joints)
        self.link_colors = make_list(link_colors, num_joints)

        # if len(link_radii)>1: self.link_radii = link_radii
        # else: self.link_radii = link_radii * self.num_joints
        # if len(joint_radii)>1: self.joint_radii = joint_radii
        # else: self.joint_radii = joint_radii * self.num_joints
        # if len(joint_colors)>1: self.joint_colors = joint_colors
        # else: self.joint_colors = joint_colors * self.num_joints
        # if len(link_lengths)>1: self.link_lengths = link_lengths
        # else: self.link_lengths = link_lengths * self.num_joints
        # if len(link_colors)>1: self.link_colors = link_colors
        # else: self.link_colors = link_colors * self.num_joints

        self.curr_vecs = np.zeros([num_joints, 3]) #rows go from proximal to distal links

        # set initial vecs to correct orientations (arm starts out vertical)
        self.curr_vecs[0,2] = self.link_lengths[0]
        self.curr_vecs[1:,0] = self.link_lengths[1:]

        # Create links for all but most distal link
        self.links = [Group((Cylinder(radius=self.link_radii[i], height=self.link_lengths[i], color=self.link_colors[i]), Sphere(radius=self.joint_radii[i],color=self.joint_colors[i]))) for i in range(self.num_joints-1)]
        
        # Add distal link as a cone instead of cylinder
        cone = Cone(radius1=self.link_radii[-1], radius2=self.link_radii[-1]/2, height=self.link_lengths[-1], color=self.link_colors[-1])
        sphere = Sphere(radius=self.joint_radii[-1],color=self.joint_colors[-1])
        distal_link = Group((cone, sphere)).translate(0, 0, self.link_lengths[-2])
        self.links.append(distal_link)

        # self.links = self.links + [distal_link]

        # self.link4 = Group((Cylinder(radius=link_radii[3], height=link_lengths[3], color=link_colors[3]), Sphere(radius=joint_radii[3],color=joint_colors[3])))
        # self.link3 = Group((Cylinder(radius=link_radii[2], height=link_lengths[2], color=link_colors[2]), Sphere(radius=joint_radii[2],color=joint_colors[2])))
        # self.link2 = Group((Cylinder(radius=link_radii[1], height=link_lengths[1], color=link_colors[1]), Sphere(radius=joint_radii[1],color=joint_colors[1])))
        # self.link1 = Group((Cone(radius1=link_radii[0], radius2 = link_radii[1]/2, height=link_lengths[0], color=link_colors[0]), Sphere(radius=joint_radii[0],color=joint_colors[0]))).translate(0,0,self.link_lengths[1])
        
        self.link_groups = [self.links[-1]]
        for i in range(1,self.num_joints-1):
            self.link_groups = self.link_groups + [Group([self.links[-i-1], self.link_groups[i-1]]).translate(0,0,self.link_lengths[-i-2])]
        self.link_groups = self.link_groups + [Group([self.links[0], self.link_groups[2]])]
        self.link_groups.reverse()

        # Call the parent constructor
        super(RobotArmGen2D, self).__init__([self.link_groups[0]], **kwargs)

        # Instantiate the kinematic chain object
        if self.num_joints == 2:
            self.kin_chain = robot_arms.PlanarXZKinematicChain2Link(link_lengths)
            self.kin_chain.joint_limits = [(-pi,pi), (-pi,0)]
        else:
            self.kin_chain = robot_arms.PlanarXZKinematicChain(link_lengths)

            # TODO the code below is (obviously) specific to a 4-joint chain
            self.kin_chain.joint_limits = [(-pi,pi), (-pi,0), (-pi/2,pi/2), (-pi/2, 10*pi/180)]

    def _update_links(self):
        self.link_groups[0].xfm.rotate = Quaternion.rotate_vecs((0,0,1), self.curr_vecs[0]).norm()
        self.link_groups[0]._recache_xfm()
        for i in range(1, self.num_joints):
            self.link_groups[i].xfm.rotate = Quaternion.rotate_vecs((1,0,0), self.curr_vecs[i]).norm()
            self.link_groups[i]._recache_xfm()

    def get_endpoint_pos(self):
        '''
        Returns the current position of the non-anchored end of the arm.
        '''
        relangs = np.arctan2(self.curr_vecs[:,2], self.curr_vecs[:,0])
        return self.perform_fk(relangs)      

    def perform_fk(self, angs):
        absvecs = np.zeros(self.curr_vecs.shape)
        for i in range(self.num_joints):
            absvecs[i] = self.link_lengths[i]*np.array([np.cos(np.sum(angs[:i+1])), 0, np.sin(np.sum(angs[:i+1]))])
        return np.sum(absvecs,axis=0)

    def set_endpoint_pos(self, pos, **kwargs):
        '''
        Positions the arm according to specified endpoint position. 
        '''
        if pos is not None:
            # Run the inverse kinematics
            angles = self.perform_ik(pos, **kwargs)

            # Update the joint configuration    
            self.set_joint_pos(angles)

    def perform_ik(self, pos, **kwargs):
        angles = self.kin_chain.inverse_kinematics(pos, q_start=-self.get_joint_pos(), verbose=False, eps=0.008, **kwargs)

        # Negate the angles. The convention in the robotics library is 
        # inverted, i.e. in the robotics library, positive is clockwise 
        # rotation whereas here CCW rotation is positive. 
        angles = -angles        
        return angles

    def calc_joint_angles(self, vecs):
        return np.arctan2(vecs[:,2], vecs[:,0])

    def get_joint_pos(self):
        '''
        Returns the joint angles of the arm in radians
        '''
        
        return self.calc_joint_angles(self.curr_vecs)
        
    def set_joint_pos(self,theta):
        '''
        Set the joint by specifying the angle in radians. Theta is a list of angles. If an element of theta = NaN, angle should remain the same.
        '''
        for i in range(self.num_joints):
            if theta[i] is not None and ~np.isnan(theta[i]):
                self.curr_vecs[i] = self.link_lengths[i]*np.array([np.cos(theta[i]), 0, np.sin(theta[i])])
                
        self._update_links()

# class RobotArm2J2D(RobotArmGen2D):
#     pass
    # def __init__(self, *args, **kwargs):
    #     super()

    # def perform_ik(self,pos):
    #     x,y,z = pos
    #     # set y to 0 for 2D
    #     y=0
    #     # if position is out of reach, set it to nearest point that can be reached
    #     mag = np.linalg.norm([x,y,z])
    #     normed = np.array([x,y,z])/mag
    #     if mag>np.sum(self.link_lengths):    
    #         x,y,z = normed*np.sum(self.link_lengths)
    #     if mag<np.abs(np.diff(self.link_lengths)):
    #         x,y,z = normed*np.abs(np.diff(self.link_lengths))

    #     return inv_kin_2D(np.array([x,y,z]), self.link_lengths[1], self.link_lengths[0])


class RobotArm2J2D(RobotArm2D):
    '''
    A 2 joint version of the 2D robot arm plant.
    '''
    def __init__(self, num_joints=2, link_radii=[.2, .2], joint_radii=[.5, .5],link_lengths=[5, 5], joint_colors = [(1,1,1,1), (1,1,1,1)],
        link_colors = [(1,1,1,1), (1,1,1,1)], **kwargs):
        self.num_joints = 2
        num_joints = self.num_joints

        # self.link_radii = link_radii
        # self.joint_radii = joint_radii
        # self.link_lengths = link_lengths
        # self.joint_colors = joint_colors
        # self.link_colors = link_colors
        link_radii = self.link_radii = make_list(link_radii, num_joints)
        joint_radii = self.joint_radii = make_list(joint_radii, num_joints)
        link_lengths = self.link_lengths = make_list(link_lengths, num_joints)
        joint_colors = self.joint_colors = make_list(joint_colors, num_joints)
        link_colors = self.link_colors = make_list(link_colors, num_joints)


        self.curr_vecs = np.zeros([2,3])

        self.curr_vecs[:,0] = self.link_lengths
        
        self.link2 = Group((Cylinder(radius=link_radii[1], height=link_lengths[1], color=link_colors[1]), Sphere(radius=joint_radii[1],color=joint_colors[1])))
        self.link1 = Group((Cone(radius1=link_radii[0], radius2 = link_radii[1]/2, height=link_lengths[0], color=link_colors[0]), Sphere(radius=joint_radii[0],color=joint_colors[0]))).translate(0,0,self.link_lengths[1])
        self.link_group_1 = Group([self.link2, self.link1])

        super(RobotArm2D, self).__init__([self.link_group_1], **kwargs)

    def _update_links(self):
        self.link_group_1.xfm.rotate = Quaternion.rotate_vecs((0,0,1),self.curr_vecs[1]).norm()
        self.link_group_1._recache_xfm()
        super(RobotArm2J2D, self)._update_links()

    def get_endpoint_pos(self):
        '''
        Returns the current position of the non-anchored end of the arm.
        '''
        relangs = np.arctan2(self.curr_vecs[:,2], self.curr_vecs[:,0])
        return self.perform_fk(relangs)      

    # def perform_fk(self, angs):
    #     abselang = np.sum(angs)
    #     abselvec = self.link_lengths[0]*np.array([np.cos(abselang), 0, np.sin(abselang)])
    #     shvec = self.link_lengths[1]*np.array([np.cos(angs[1]), 0, np.sin(angs[1])])
    #     return abselvec + shvec

    def perform_fk(self, angs):
        absvecs = np.zeros(self.curr_vecs.shape)
        for i in range(self.num_joints):
            absvecs[i] = self.link_lengths[i]*np.array([np.cos(np.sum(angs[:i+1])), 0, np.sin(np.sum(angs[:i+1]))])
        return np.sum(absvecs,axis=0)        

    def set_endpoint_pos(self, pos, **kwargs):
        '''
        Positions the arm according to specified endpoint position. Uses 2D inverse kinematic equations to calculate joint angles.
        '''
        if pos is not None:
            angles = self.perform_ik(pos)
            self.set_joint_pos([angles['sh_pabd'], angles['el_pflex']])

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
