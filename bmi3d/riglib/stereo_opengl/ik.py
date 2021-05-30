'''
This module implements IK functions for running inverse kinematics.
Current, only a two-joint system can be modelled.
'''

import numpy as np

from .xfm import Quaternion
from .models import Group
from .primitives import Cylinder, Sphere, Cone
from ..bmi import robot_arms

pi = np.pi

joint_angles_dtype = [('sh_pflex', np.float64), ('sh_pabd', np.float64), ('sh_prot', np.float64), ('el_pflex', np.float64), ('el_psup', np.float64)]
joint_vel_dtype = [('sh_vflex', np.float64), ('sh_vabd', np.float64), ('sh_vrot', np.float64), ('el_vflex', np.float64), ('el_vsup', np.float64)]

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
        pass

    if vel is not None:
        joint_vel = np.zeros(len(pos), dtype=joint_vel_dtype)
        
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

class Plant(object):
    def __init__(self, *args, **kwargs):
        super(Plant, self).__init__(*args, **kwargs)

    def drive(self, decoder):
        self.set_intrinsic_coordinates(decoder['q'])
        intrinsic_coords = self.get_intrinsic_coordinates()
        if not np.any(np.isnan(intrinsic_coords)):
            decoder['q'] = self.get_intrinsic_coordinates()        


class CursorPlant(Plant):
    def __init__(self, endpt_bounds=None, **kwargs):
        self.endpt_bounds = endpt_bounds
        self.position = np.array([0., 0., 0.]) #np.zeros(3)

    def get_endpoint_pos(self):
        return self.position

    def set_endpoint_pos(self, pt, **kwargs):
        self.position = pt

    def get_intrinsic_coordinates(self):
        return self.position

    def set_intrinsic_coordinates(self, pt):
        self.position = pt

    def drive(self, decoder):
        pos = decoder['q'].copy()
        vel = decoder['qdot'].copy()
        
        if self.endpt_bounds is not None:
            if pos[0] < self.endpt_bounds[0]: 
                pos[0] = self.endpt_bounds[0]
                #vel[0] = 0
            if pos[0] > self.endpt_bounds[1]: 
                pos[0] = self.endpt_bounds[1]
                #vel[0] = 0

            if pos[1] < self.endpt_bounds[2]: 
                pos[1] = self.endpt_bounds[2]
                #vel[1] = 0
            if pos[1] > self.endpt_bounds[3]: 
                pos[1] = self.endpt_bounds[3]
                #vel[1] = 0

            if pos[2] < self.endpt_bounds[4]: 
                pos[2] = self.endpt_bounds[4]
                #vel[2] = 0
            if pos[2] > self.endpt_bounds[5]: 
                pos[2] = self.endpt_bounds[5]
                #vel[2] = 0
        
        decoder['q'] = pos
        decoder['qdot'] = vel
        super(CursorPlant, self).drive(decoder)


class RobotArmGen2D(Plant, Group):
    def __init__(self, link_radii=arm_radius, joint_radii=arm_radius, link_lengths=[15,15,5,5], joint_colors=arm_color,
        link_colors=arm_color, base_loc=np.array([2., 0., -15]), **kwargs):
        '''
        Instantiate the graphics and the virtual arm for a planar kinematic chain
        '''
        num_joints = len(link_lengths)
        self.num_joints = num_joints

        self.link_radii = make_list(link_radii, num_joints)
        self.joint_radii = make_list(joint_radii, num_joints)
        self.link_lengths = make_list(link_lengths, num_joints)
        self.joint_colors = make_list(joint_colors, num_joints)
        self.link_colors = make_list(link_colors, num_joints)

        self.curr_vecs = np.zeros([num_joints, 3]) #rows go from proximal to distal links

        # set initial vecs to correct orientations (arm starts out vertical)
        self.curr_vecs[0,2] = self.link_lengths[0]
        self.curr_vecs[1:,0] = self.link_lengths[1:]

        # Create links
        self.links = []

        for i in range(self.num_joints):
            joint = Sphere(radius=self.joint_radii[i], color=self.joint_colors[i])

            # The most distal link gets a tapered cylinder (for purely stylistic reasons)
            if i < self.num_joints - 1:
                link = Cylinder(radius=self.link_radii[i], height=self.link_lengths[i], color=self.link_colors[i])
            else:
                link = Cone(radius1=self.link_radii[-1], radius2=self.link_radii[-1]/2, height=self.link_lengths[-1], color=self.link_colors[-1])
            link_i = Group((link, joint))
            self.links.append(link_i)

        link_offsets = [0] + self.link_lengths[:-1]
        self.link_groups = [None]*self.num_joints
        for i in range(self.num_joints)[::-1]:
            if i == self.num_joints-1:
                self.link_groups[i] = self.links[i]
            else:
                self.link_groups[i] = Group([self.links[i], self.link_groups[i+1]])

            self.link_groups[i].translate(0, 0, link_offsets[i])

        # Call the parent constructor
        super(RobotArmGen2D, self).__init__([self.link_groups[0]], **kwargs)

        # Instantiate the kinematic chain object
        if self.num_joints == 2:
            self.kin_chain = robot_arms.PlanarXZKinematicChain(link_lengths)
            self.kin_chain.joint_limits = [(-pi,pi), (-pi,0)]
        else:
            self.kin_chain = robot_arms.PlanarXZKinematicChain(link_lengths)

            # TODO the code below is (obviously) specific to a 4-joint chain
            self.kin_chain.joint_limits = [(-pi,pi), (-pi,0), (-pi/2,pi/2), (-pi/2, 10*pi/180)]

        self.base_loc = base_loc
        self.translate(*self.base_loc, reset=True)

    def _update_links(self):
        for i in range(0, self.num_joints):
            # Rotate each joint to the vector specified by the corresponding row in self.curr_vecs
            # Annoyingly, the baseline orientation of the first group is always different from the 
            # more distal attachments, so the rotations have to be found relative to the orientation 
            # established at instantiation time.
            if i == 0:
                baseline_orientation = (0, 0, 1)
            else:
                baseline_orientation = (1, 0, 0)

            # Find the normalized quaternion that represents the desired joint rotation
            self.link_groups[i].xfm.rotate = Quaternion.rotate_vecs(baseline_orientation, self.curr_vecs[i]).norm()

            # Recompute any cached transformations after the change
            self.link_groups[i]._recache_xfm()

    def get_endpoint_pos(self):
        '''
        Returns the current position of the non-anchored end of the arm.
        '''
        relangs = np.arctan2(self.curr_vecs[:,2], self.curr_vecs[:,0])
        return self.perform_fk(relangs) + self.base_loc

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
            self.set_intrinsic_coordinates(angles)

    def perform_ik(self, pos, **kwargs):
        angles = self.kin_chain.inverse_kinematics(pos - self.base_loc, q_start=-self.get_intrinsic_coordinates(), verbose=False, eps=0.008, **kwargs)
        # print self.kin_chain.endpoint_pos(angles)

        # Negate the angles. The convention in the robotics library is 
        # inverted, i.e. in the robotics library, positive is clockwise 
        # rotation whereas here CCW rotation is positive. 
        angles = -angles        
        return angles

    def calc_joint_angles(self, vecs):
        return np.arctan2(vecs[:,2], vecs[:,0])

    def get_intrinsic_coordinates(self):
        '''
        Returns the joint angles of the arm in radians
        '''
        
        return self.calc_joint_angles(self.curr_vecs)
        
    def set_intrinsic_coordinates(self,theta):
        '''
        Set the joint by specifying the angle in radians. Theta is a list of angles. If an element of theta = NaN, angle should remain the same.
        '''
        for i in range(self.num_joints):
            if theta[i] is not None and ~np.isnan(theta[i]):
                self.curr_vecs[i] = self.link_lengths[i]*np.array([np.cos(theta[i]), 0, np.sin(theta[i])])
                
        self._update_links()


class RobotArmGen3D(Plant, Group):
    def __init__(self, link_radii=arm_radius, joint_radii=arm_radius, link_lengths=[15,15,5,5], joint_colors=arm_color,
        link_colors=arm_color, base_loc=np.array([2., 0., -15]), **kwargs):
        '''
        Instantiate the graphics and the virtual arm for a kinematic chain
        '''
        num_joints = 2
        self.num_joints = 2

        self.link_radii = make_list(link_radii, num_joints)
        self.joint_radii = make_list(joint_radii, num_joints)
        self.link_lengths = make_list(link_lengths, num_joints)
        self.joint_colors = make_list(joint_colors, num_joints)
        self.link_colors = make_list(link_colors, num_joints)

        self.curr_vecs = np.zeros([num_joints, 3]) #rows go from proximal to distal links

        # set initial vecs to correct orientations (arm starts out vertical)
        self.curr_vecs[0,2] = self.link_lengths[0]
        self.curr_vecs[1:,0] = self.link_lengths[1:]

        # Create links
        self.links = []

        for i in range(self.num_joints):
            joint = Sphere(radius=self.joint_radii[i], color=self.joint_colors[i])

            # The most distal link gets a tapered cylinder (for purely stylistic reasons)
            if i < self.num_joints - 1:
                link = Cylinder(radius=self.link_radii[i], height=self.link_lengths[i], color=self.link_colors[i])
            else:
                link = Cone(radius1=self.link_radii[-1], radius2=self.link_radii[-1]/2, height=self.link_lengths[-1], color=self.link_colors[-1])
            link_i = Group((link, joint))
            self.links.append(link_i)

        link_offsets = [0] + self.link_lengths[:-1]
        self.link_groups = [None]*self.num_joints
        for i in range(self.num_joints)[::-1]:
            if i == self.num_joints-1:
                self.link_groups[i] = self.links[i]
            else:
                self.link_groups[i] = Group([self.links[i], self.link_groups[i+1]])

            self.link_groups[i].translate(0, 0, link_offsets[i])

        # Call the parent constructor
        super(RobotArmGen3D, self).__init__([self.link_groups[0]], **kwargs)

        # Instantiate the kinematic chain object
        if self.num_joints == 2:
            self.kin_chain = robot_arms.PlanarXZKinematicChain(link_lengths)
            self.kin_chain.joint_limits = [(-pi,pi), (-pi,0)]
        else:
            self.kin_chain = robot_arms.PlanarXZKinematicChain(link_lengths)

            # TODO the code below is (obviously) specific to a 4-joint chain
            self.kin_chain.joint_limits = [(-pi,pi), (-pi,0), (-pi/2,pi/2), (-pi/2, 10*pi/180)]

        self.base_loc = base_loc
        self.translate(*self.base_loc, reset=True)

    def _update_links(self):
        for i in range(0, self.num_joints):
            # Rotate each joint to the vector specified by the corresponding row in self.curr_vecs
            # Annoyingly, the baseline orientation of the first group is always different from the 
            # more distal attachments, so the rotations have to be found relative to the orientation 
            # established at instantiation time.
            if i == 0:
                baseline_orientation = (0, 0, 1)
            else:
                baseline_orientation = (1, 0, 0)

            # Find the normalized quaternion that represents the desired joint rotation
            self.link_groups[i].xfm.rotate = Quaternion.rotate_vecs(baseline_orientation, self.curr_vecs[i]).norm()

            # Recompute any cached transformations after the change
            self.link_groups[i]._recache_xfm()

    def get_endpoint_pos(self):
        '''
        Returns the current position of the non-anchored end of the arm.
        '''
        relangs_xz = np.arctan2(self.curr_vecs[:,2], self.curr_vecs[:,0])
        relangs_xy = np.arctan2(self.curr_vecs[:,1], self.curr_vecs[:,0])
        return self.perform_fk(relangs_xz, relangs_xy) + self.base_loc

    def perform_fk(self, angs_xz, angs_xy):
        absvecs = np.zeros(self.curr_vecs.shape)
        for i in range(self.num_joints):
            absvecs[i] = self.link_lengths[i]*np.array([np.cos(np.sum(angs_xz[:i+1])), np.sin(np.sum(angs_xy[:i+1])), np.sin(np.sum(angs_xz[:i+1]))])
        return np.sum(absvecs,axis=0)

    def set_endpoint_pos(self, pos, **kwargs):
        '''
        Positions the arm according to specified endpoint position. 
        '''
        if pos is not None:
            # Run the inverse kinematics
            angles = self.perform_ik(pos, **kwargs)

            # Update the joint configuration    
            self.set_intrinsic_coordinates(angles)


    def perform_ik(self, pos, **kwargs):
        angles = self.kin_chain.inverse_kinematics(pos - self.base_loc, q_start=-self.get_intrinsic_coordinates(), verbose=False, eps=0.008, **kwargs)
        # print self.kin_chain.endpoint_pos(angles)

        # Negate the angles. The convention in the robotics library is 
        # inverted, i.e. in the robotics library, positive is clockwise 
        # rotation whereas here CCW rotation is positive. 
        angles = -angles




        '''Sets the endpoint coordinate for the two-joint system'''
        #Make sure the target is actually achievable
        if np.linalg.norm(pos) > self.tlen:
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
        # print self.upperarm.xfm.rotate
        self.curr_vecs[0] = self.lengths[0]*self.upperarm.xfm.rotate.quat[1:]
        self.curr_vecs[1] = self.lengths[1]*self.forearm.xfm.rotate.quat[1:]
        print(self.forearm.xfm)
        print(self.upperarm.xfm)
        # raise NotImplementedError("update curr_vecs!")

              
        return angles

    def calc_joint_angles(self, vecs):
        return np.arctan2(vecs[:,2], vecs[:,0]), np.arctan2(vecs[:,1], vecs[:,0])

    def get_intrinsic_coordinates(self):
        '''
        Returns the joint angles of the arm in radians
        '''
        
        return self.calc_joint_angles(self.curr_vecs)
        
    def set_intrinsic_coordinates(self,theta_xz, theta_xy):
        '''
        Set the joint by specifying the angle in radians. Theta is a list of angles. If an element of theta = NaN, angle should remain the same.
        '''
        for i in range(self.num_joints):
            if theta_xz[i] is not None and ~np.isnan(theta_xz[i]) and theta_xy[i] is not None and ~np.isnan(theta_xy[i]):
                self.curr_vecs[i] = self.link_lengths[i]*np.array([np.cos(theta_xz[i]), np.sin(theta_xy[i]), np.sin(theta_xz[i])])
                
        self._update_links()

    def _midpos(self, target):
            m, n = self.link_lengths
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


class RobotArm2J2D(RobotArmGen2D):
    def drive(self, decoder):
        raise NotImplementedError("deal with the state bounding stuff!")
        # elif self.decoder.ssm == train.joint_2D_state_space:
        #     self.set_arm_joints(self.decoder['sh_pabd', 'el_pflex'])

        #     # Force the arm to a joint configuration where the cursor is on-screen
        #     pos = self.get_arm_endpoint()
        #     pos = self.apply_cursor_bounds(pos)
        #     self.set_arm_endpoint(pos)

        #     # Reset the decoder state to match the joint configuration of the arm
        #     joint_pos = self.get_arm_joints()
        #     self.decoder['sh_pabd', 'el_pflex'] = joint_pos

class TwoJoint(object):
    '''
    Models a two-joint IK system (arm, leg, etc). Constrains the system by 
    always having middle joint halfway between the origin and target
    '''
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
        # print self.upperarm.xfm.rotate
        upperarm_affine_xform = self.upperarm.xfm.rotate.to_mat()
        forearm_affine_xform = (self.upperarm.xfm * self.forearm.xfm).rotate.to_mat()
        # print np.dot(upperarm_affine_xform, np.array([0., 0, self.lengths[0], 1]))
        self.curr_vecs[0] = np.dot(upperarm_affine_xform, np.array([0., 0, self.lengths[0], 1]))[:-1]#self.lengths[0]*self.upperarm.xfm.rotate.quat[1:]
        self.curr_vecs[1] = np.dot(forearm_affine_xform, np.array([0, 0, self.lengths[1], 1]))[:-1]#self.lengths[1]*self.forearm.xfm.rotate.quat[1:]
        # print self.forearm.xfm
        # print self.upperarm.xfm
        # raise NotImplementedError("update curr_vecs!")

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


        # cursor_color = (.5,0,.5,1)
        # cursor_radius = 0.4
        # self.endpt_cursor = Sphere(radius=cursor_radius, color=cursor_color)
        # self.endpt_cursor.translate(0, 0, lengths[1])
        # self.forearm = Group([
        #     Cylinder(radius=link_radii[1], height=lengths[1], color=link_colors[1]), 
        #     self.endpt_cursor])
        # self.forearm.translate(0,0,lengths[0])
        
        # self.upperarm = Group([
        #     Cylinder(radius=link_radii[0], height=lengths[0],color=link_colors[0]), 
        #     Sphere(radius=ball_radii[0],color=ball_colors[0]).translate(0, 0, lengths[0]),
        #     self.forearm])
        # self.system = TwoJoint(self.upperarm, self.forearm, lengths = (self.lengths))

class RobotArm(Plant, Group):
    def __init__(self, link_radii=(.2, .2), ball_radii=(.5,.5),lengths=(20, 20), ball_colors = ((1,1,1,1),(1,1,1,1)),\
        link_colors = ((1,1,1,1), (1,1,1,1)), base_loc=np.array([2., 0., -10.]), **kwargs):
        self.link_radii = link_radii
        self.ball_radii = ball_radii
        self.lengths = lengths

        self.endpt_cursor = Sphere(radius=ball_radii[1], color=(1, 0, 1, 1)) #ball_colors[1])
        self.forearm = Group([
            Cylinder(radius=link_radii[1], height=lengths[1], color=link_colors[1]), 
            self.endpt_cursor.translate(0, 0, lengths[1])]).translate(0,0,lengths[0])
        self.upperarm = Group([
            Cylinder(radius=link_radii[0], height=lengths[0],color=link_colors[0]), 
            Sphere(radius=ball_radii[0],color=ball_colors[0]).translate(0, 0, lengths[0]),
            self.forearm])
        self.system = TwoJoint(self.upperarm, self.forearm, lengths = (self.lengths))
        super(RobotArm, self).__init__([self.upperarm], **kwargs)

        self.num_links = len(link_radii)
        self.num_joints = 3 # abstract joints. this system is fully characterized by the endpoint position since the elbow angle is determined by IK

        self.base_loc = base_loc

        self.translate(*self.base_loc, reset=True)

    def get_endpoint_pos(self):
        # print 'curr_vecs', self.system.curr_vecs
        # print
        return np.sum(self.system.curr_vecs, axis=0) + self.base_loc

    def set_endpoint_pos(self, pos, **kwargs):
        self.system.set_endpoint_3D(pos - self.base_loc)

    def get_intrinsic_coordinates(self):
        return self.get_endpoint_pos()

    def set_intrinsic_coordinates(self, pos):
        self.set_endpoint_pos(pos)

    # def drive(self, decoder):
    #     self.set_intrinsic_coordinates(decoder['q'])
    #     intrinsic_coords = self.get_intrinsic_coordinates()
    #     if not np.any(np.isnan(intrinsic_coords)):
    #         decoder['q'] = self.get_intrinsic_coordinates()
    #     print 'arm pos', self.get_endpoint_pos()

    # def set_endpoint_2D(self, target):
    #     self.system.set_endpoint_2D(target)

    # def set_joints_2D(self, shoulder_angle, elbow_angle): 
    #     self.system.set_joints_2D(shoulder_angle, elbow_angle)

    # def get_hand_location(self, shoulder_anchor):
    #     ''' returns position of ball at end of forearm (hand)'''
    #     return shoulder_anchor + self.system.curr_vecs[0] +self.system.curr_vecs[1]

    # def get_joint_angles_2D(self):
    #     return self.system.curr_angles[0], self.system.curr_angles[1] - np.pi/2





# cursor_bounds = traits.Tuple((-25, 25, 0, 0, -14, 14), "Boundaries for where the cursor can travel on the screen")

chain_kwargs = dict(link_radii=.6, joint_radii=0.6, joint_colors=(181/256., 116/256., 96/256., 1), link_colors=(181/256., 116/256., 96/256., 1))

shoulder_anchor = np.array([2., 0., -15])

chain_15_15_5_5 = RobotArmGen2D(link_lengths=[15, 15, 5, 5], base_loc=shoulder_anchor, **chain_kwargs)
init_joint_pos = np.array([ 0.47515737,  1.1369006 ,  1.57079633,  0.29316668])  ## center pos coordinates: 0.63017,  1.38427,  1.69177,  0.42104 
chain_15_15_5_5.set_intrinsic_coordinates(init_joint_pos)

chain_20_20 = RobotArm2J2D(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)
starting_pos = np.array([5., 0., 5])
chain_20_20.set_endpoint_pos(starting_pos - shoulder_anchor, n_iter=10, n_particles=500)
chain_20_20.set_endpoint_pos(starting_pos, n_iter=10, n_particles=500)

cursor = CursorPlant(endpt_bounds=(-14, 14, 0., 0., -14, 14))
#cursor = CursorPlant(endpt_bounds=(-10, 10, 0., 0., -10, 10))
#cursor = CursorPlant(endpt_bounds=(-9.5, 9.5, 0., 0., -7.5, 11.5))
#cursor = CursorPlant(endpt_bounds=(-11, 11., 0., 0., -11., 11.))
#cursor = CursorPlant(endpt_bounds=(-10, 10., 0., 0., -10., 10.))
#cursor = CursorPlant(endpt_bounds=(-24., 24., 0., 0., -14., 14.))

arm_3d = RobotArm()

plants = dict(RobotArmGen2D=chain_15_15_5_5, 
              RobotArm2J2D=chain_20_20,
              CursorPlant=cursor,
              Arm3D=arm_3d)
