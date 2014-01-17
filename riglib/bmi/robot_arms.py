import numpy as np
import robot

import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 10

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import izip

import time

pi = np.pi

class KinematicChain(object):
    '''
    Arbitrary kinematic chain (i.e. spherical joint at the beginning of 
    each joint)
    '''
    def __init__(self, link_lengths=[10., 10.], name=''):
        self.n_links = len(link_lengths)
        self.link_lengths = link_lengths

        links = []
        for link_length in link_lengths:
            link1 = robot.Link(alpha=-pi/2)
            link2 = robot.Link(alpha=pi/2)
            link3 = robot.Link(d=-link_length)
            links += [link1, link2, link3]
        
        # By convention, we start the arm in the XY-plane
        links[1].offset = pi/2 

        self.robot = robot.SerialLink(links)
        self.robot.name = name

    def calc_full_joint_angles(self, joint_angles):
        return joint_angles

    def full_angles_to_subset(self, joint_angles):
        return joint_angles

    def plot(self, joint_angles):
        joint_angles = self.calc_full_joint_angles(joint_angles)
        self.robot.plot(joint_angles)

    def forward_kinematics(self, joint_angles):
        '''
        Calculate forward kinematics using D-H parameter convention
        '''
        joint_angles = self.calc_full_joint_angles(joint_angles)
        t, allt = self.robot.fkine(joint_angles)
        self.joint_angles = joint_angles
        self.t = t
        self.allt = allt
        return t, allt

    def apply_joint_limits(self, joint_angles):
        return joint_angles

    def inverse_kinematics(self, starting_config, target_pos, n_iter=1000, 
                           verbose=False, eps=0.1, return_path=False):
        '''
        Default inverse kinematics method is RRT since for redundant 
        kinematic chains, an infinite number of inverse kinematics solutions 
        exist
        '''

        q = starting_config
        start_time = time.time()
        n_iter = 1000
        endpoint_traj = np.zeros([n_iter, 3])
        for k in range(n_iter):
            # calc endpoint position of the manipulator
            endpoint_traj[k] = self.endpoint_pos(q)

            if np.linalg.norm(endpoint_traj[k] - target_pos) < eps:
                break

            # calculate the jacobian
            J = self.jacobian(q)
            J_pos = J[0:3,:]

            # take a step from the current position toward the target pos using the inverse Jacobian
            # J_inv = np.linalg.pinv(J_pos)
            J_inv = J_pos.T

            # xdot = (endpoint_traj[k] - target_pos)/np.linalg.norm(endpoint_traj[k] - target_pos)
            xdot = (target_pos - endpoint_traj[k])/np.linalg.norm(endpoint_traj[k] - target_pos)
            qdot = 0.001*np.dot(J_inv, xdot)
            qdot = self.full_angles_to_subset(np.array(qdot).ravel())

            q += qdot

            # apply joint limits
            q = self.apply_joint_limits(q)

        end_time = time.time()
        runtime = end_time - start_time
        if verbose:
            print "Runtime: %g" % runtime

        if return_path:
            return q, endpoint_traj
        else:
            return q

    def jacobian(self, joint_angles):
        joint_angles = self.calc_full_joint_angles(joint_angles)
        J = self.robot.jacobn(joint_angles)
        return J

    def endpoint_pos(self, joint_angles):
        t, allt = self.forward_kinematics(joint_angles)
        return np.array(t[0:3,-1]).ravel()

class PlanarXZKinematicChain(KinematicChain):
    '''
    Kinematic chain restricted to movement in the XZ-plane
    '''
    def calc_full_joint_angles(self, joint_angles):
        '''
        only some joints rotate in the planar kinematic chain

        '''
        if not len(joint_angles) == self.n_links:
            raise ValueError("Incorrect number of joint angles specified!")

        # There are really 3 angles per joint to allow 3D rotation at each joint
        joint_angles_full = np.zeros(self.n_links * 3)  
        joint_angles_full[1::3] = joint_angles
        return joint_angles_full 

    def full_angles_to_subset(self, joint_angles):
        return joint_angles[1::3]

    def apply_joint_limits(self, joint_angles):
        if not hasattr(self, 'joint_limits'):
            return joint_angles
        else:
            angles = []
            for angle, (lim_min, lim_max) in izip(joint_angles, self.joint_limits):
                angle = max(lim_min, angle)
                angle = min(angle, lim_max)
                angles.append(angle)

            return np.array(angles)


    # def inverse_kinematics(self, starting_config, endpoint_pos):
    #     x, y, z = endpoint_pos
    #     if not y == 0:
    #         raise ValueError("PlanarXZKinematicChain requires y=0")
    #     return super(PlanarXZKinematicChain, self).inverse_kinematics(starting_config, endpoint_pos)


class RobotArm(object):
    def __init__(self, forearm_length=14.5, upper_arm_length=14.5):
        self.forearm_length = forearm_length
        self.upper_arm_length = upper_arm_length

        link1 = robot.Link(theta=pi, alpha=-pi/2)
        link2 = robot.Link(theta=pi, alpha=pi/2)
        link2.offset = -pi/2
        link3 = robot.Link(d=-upper_arm_length, theta=pi, alpha=-pi/2);
        link4 = robot.Link(theta=pi, alpha=pi/2);
        link5 = robot.Link(d=-forearm_length);

        self.robot = robot.SerialLink([link1, link2, link3, link4, link5]);
        self.robot.name = 'exo'

    def forward_kinematics(self, joint_angles):
        '''
        Calculate forward kinematics using D-H parameter convention
        '''
        t, allt = self.robot.fkine(joint_angles)
        self.joint_angles = joint_angles
        self.t = t
        self.allt = allt
        return t, allt

    def inverse_kinematics(self):
        '''
        Inverse kinematics for redundant exoskeleton
        '''
        raise NotImplementedError



class RobotArm2D(RobotArm):
    def inverse_kinematics(self, pos):
        '''
        Inverse kinematics for a 2D arm. This function returns all 5 angles required
        to specify the pose of the exoskeleton (see riglib.bmi.train for the 
        definitions of these angles). This pose is constrained to the x-z plane
        by forcing shoulder flexion/extension, elbow rotation and supination/pronation
        to always be 0. 
        '''
        l_upperarm = self.upper_arm_length
        l_forearm = self.forearm_length
        # require the y-coordinate to be 0, i.e. flat on the screen
        x, y, z = pos
        assert y == 0
        L = np.sqrt(x**2 + z**2)
        cos_el_pflex = (L**2 - l_forearm**2 + l_upperarm**2) / (2*l_forearm*l_upperarm)
    
        angles = OrderedDict()
        angles['sh_pflex'] = 0
        angles['sh_pabd'] = np.atan(z/x) - np.asin((l_forearm * cos_el_pflex) / L)
        angles['el_pflex'] = np.acos(cos_el_pflex)
        angles['el_prot'] = 0
        angles['el_psup'] = 0
        return angles
