#!/usr/bin/python
'''
Representations of plants (control systems)
'''
import numpy as np
from stereo_opengl.primitives import Cylinder, Sphere, Cone
from stereo_opengl.models import Group
from riglib.bmi import robot_arms
from riglib.stereo_opengl.xfm import Quaternion

class Plant(object):
    hdf_attrs = []
    def __init__(self):
        raise NotImplementedError

    def drive(self, decoder):
        self.set_intrinsic_coordinates(decoder['q'])
        intrinsic_coords = self.get_intrinsic_coordinates()
        if not np.any(np.isnan(intrinsic_coords)):
            decoder['q'] = self.get_intrinsic_coordinates()

    def get_data_to_save(self):
        raise NotImplementedError

class CursorPlant(Plant):
    hdf_attrs = [('cursor', 'f8', (3,))]
    def __init__(self, endpt_bounds=None, cursor_radius=0.4, cursor_color=(.5, 0, .5, 1), starting_pos=np.array([0., 0., 0.]), **kwargs):
        self.endpt_bounds = endpt_bounds
        self.position = starting_pos
        self.starting_pos = starting_pos
        self.cursor_radius = cursor_radius
        self.cursor_color = cursor_color
        from riglib.bmi import state_space_models
        self.ssm = state_space_models.StateSpaceEndptVel2D()
        self._pickle_init()

    def _pickle_init(self):
        self.cursor = Sphere(radius=self.cursor_radius, color=self.cursor_color)
        self.cursor.translate(*self.position, reset=True)
        self.graphics_models = [self.cursor]

    def draw(self):
        self.cursor.translate(*self.position, reset=True)

    def get_endpoint_pos(self):
        return self.position

    def set_endpoint_pos(self, pt, **kwargs):
        self.position = pt
        self.draw()

    def get_intrinsic_coordinates(self):
        return self.position

    def set_intrinsic_coordinates(self, pt):
        self.position = pt
        self.draw()

    def drive(self, decoder):
        pos = decoder['q'].copy()
        vel = decoder['qdot'].copy()
        
        if self.endpt_bounds is not None:
            if pos[0] < self.endpt_bounds[0]: 
                pos[0] = self.endpt_bounds[0]
                vel[0] = 0
            if pos[0] > self.endpt_bounds[1]: 
                pos[0] = self.endpt_bounds[1]
                vel[0] = 0

            if pos[1] < self.endpt_bounds[2]: 
                pos[1] = self.endpt_bounds[2]
                vel[1] = 0
            if pos[1] > self.endpt_bounds[3]: 
                pos[1] = self.endpt_bounds[3]
                vel[1] = 0

            if pos[2] < self.endpt_bounds[4]: 
                pos[2] = self.endpt_bounds[4]
                vel[2] = 0
            if pos[2] > self.endpt_bounds[5]: 
                pos[2] = self.endpt_bounds[5]
                vel[2] = 0
        
        decoder['q'] = pos
        decoder['qdot'] = vel
        super(CursorPlant, self).drive(decoder)

    def get_data_to_save(self):
        return dict(cursor=self.position)



class VirtualKinematicChain(Plant):
    def __init__(self, *args, **kwargs):
        
        super(VirtualKinematicChain, self).__init__(*args, **kwargs)


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
pi = np.pi
class RobotArmGen2D(Plant):
    def __init__(self, link_radii=arm_radius, joint_radii=arm_radius, link_lengths=[15,15,5,5], joint_colors=arm_color,
        link_colors=arm_color, base_loc=np.array([2., 0., -15]), joint_limits=[(-pi,pi), (-pi,0), (-pi/2,pi/2), (-pi/2, 10*pi/180)], **kwargs):
        '''
        Instantiate the graphics and the virtual arm for a planar kinematic chain
        '''
        self.num_joints = num_joints = len(link_lengths)

        self.link_radii = make_list(link_radii, num_joints)
        self.joint_radii = make_list(joint_radii, num_joints)
        self.link_lengths = make_list(link_lengths, num_joints)
        self.joint_colors = make_list(joint_colors, num_joints)
        self.link_colors = make_list(link_colors, num_joints)

        self.curr_vecs = np.zeros([num_joints, 3]) #rows go from proximal to distal links

        # set initial vecs to correct orientations (arm starts out vertical)
        self.curr_vecs[0,2] = self.link_lengths[0]
        self.curr_vecs[1:,0] = self.link_lengths[1:]

        # Instantiate the kinematic chain object
        self.kin_chain = robot_arms.PlanarXZKinematicChain(link_lengths)
        self.kin_chain.joint_limits = joint_limits

        self.base_loc = base_loc
        self._pickle_init()

        from riglib.bmi import state_space_models
        self.ssm = state_space_models.StateSpaceFourLinkTentacle2D()

        self.hdf_attrs = [('cursor', 'f8', (3,)), ('joint_angles','f8', (self.num_joints, )), ('arm_visible','f8',(1,))]

    def _pickle_init(self):
        '''
        Create graphics models
        '''
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

        self.link_groups[0].translate(*self.base_loc, reset=True)
        self.graphics_models = [self.link_groups[0]]

    def _update_link_graphics(self):
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
                
        self._update_link_graphics()

    def get_data_to_save(self):
        return dict(cursor=self.get_endpoint_pos(), joint_angles=self.get_intrinsic_coordinates())