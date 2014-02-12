import numpy as np
import robot #!!!!!!!!!!!
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
        links[1].offset = -pi/2 

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

    def inverse_kinematics(self, target_pos, q_start=None, method='pso', **kwargs):
        if q_start == None:
            q_start = self.random_sample()
        return self.inverse_kinematics_pso(target_pos, q_start, **kwargs)
        # ik_method = getattr(self, 'inverse_kinematics_%s' % method)
        # return ik_method(q_start, target_pos)

    def inverse_kinematics_grad_descent(self, target_pos, starting_config, n_iter=1000, 
                           verbose=False, eps=0.01, return_path=False):
        '''
        Default inverse kinematics method is RRT since for redundant 
        kinematic chains, an infinite number of inverse kinematics solutions 
        exist
        '''

        q = starting_config
        start_time = time.time()
        endpoint_traj = np.zeros([n_iter, 3])

        joint_limited = np.zeros(len(q))

        for k in range(n_iter):
            # print k
            # calc endpoint position of the manipulator
            endpoint_traj[k] = self.endpoint_pos(q)

            current_cost = np.linalg.norm(endpoint_traj[k] - target_pos, 2)
            if current_cost < eps:
                print "Terminating early"
                break

            # calculate the jacobian
            J = self.jacobian(q)
            J_pos = J[0:3,:]

            # for joints that are at their limit, zero out the jacobian?
            # J_pos[:, np.nonzero(self.calc_full_joint_angles(joint_limited))] = 0

            # take a step from the current position toward the target pos using the inverse Jacobian
            J_inv = np.linalg.pinv(J_pos)
            # J_inv = J_pos.T

            xdot = (target_pos - endpoint_traj[k])#/np.linalg.norm(endpoint_traj[k] - target_pos) 

            # if current_cost < 3 or k > 10:
            #     stepsize = 0.001
            # else:
            #     stepsize = 0.01


            xdot = (target_pos - endpoint_traj[k])#/np.linalg.norm(endpoint_traj[k] - target_pos)
            # xdot = (endpoint_traj[k] - target_pos)/np.linalg.norm(endpoint_traj[k] - target_pos)
            qdot = 0.001*np.dot(J_inv, xdot)
            qdot = self.full_angles_to_subset(np.array(qdot).ravel())

            q += qdot

            # apply joint limits
            q, joint_limited = self.apply_joint_limits(q)

        end_time = time.time()
        runtime = end_time - start_time
        if verbose:
            print "Runtime: %g" % runtime
            print "# of iterations: %g" % k

        if return_path:
            return q, endpoint_traj[:k]
        else:
            return q

    def jacobian(self, joint_angles):
        joint_angles = self.calc_full_joint_angles(joint_angles)
        J = self.robot.jacobn(joint_angles)
        return J

    def endpoint_pos(self, joint_angles):
        t, allt = self.forward_kinematics(joint_angles)
        return np.array(t[0:3,-1]).ravel()

    def random_sample(self):
        q_start = []
        for lim_min, lim_max in self.joint_limits:
            q_start.append(np.random.uniform(lim_min, lim_max))
        return np.array(q_start)

    def ik_cost(self, q, q_start, target_pos, weight=100):
        q_diff = q - q_start
        return np.linalg.norm(q_diff[0:2]) + weight*np.linalg.norm(self.endpoint_pos(q) - target_pos)

    def inverse_kinematics_pso(self, target_pos, q_start, time_limit=np.inf, verbose=False, eps=0.5, n_particles=10, n_iter=10):
        # Initialize the particles; 
        n_joints = self.n_joints

        particles_q = np.tile(q_start, [n_particles, 1])

        # if 0:
        #     # initialize the velocities to be biased around the direction the jacobian tells you is correct
        #     current_pos = self.endpoint_pos(q_start)
        #     int_displ = target_pos - current_pos
        #     print int_displ, target_pos
        #     J = self.jacobian(q_start)
        #     endpoint_vel = np.random.randn(n_particles, 3)# + int_displ
        #     particles_v = np.dot(J[0:3,1::3].T, endpoint_vel.T).T
        # else:
        #     # initialize particle velocities randomly

        
        particles_v = np.random.randn(n_particles, n_joints) #/ np.array([1., 1., 1, 1]) #np.array(self.link_lengths)

        cost_fn = lambda q: self.ik_cost(q, q_start, target_pos)

        gbest = particles_q.copy()
        gbestcost = np.array(map(cost_fn, gbest))
        pbest = gbest[np.argmin(gbestcost)]
        pbestcost = cost_fn(pbest)

        min_limits = np.array([x[0] for x in self.joint_limits])
        max_limits = np.array([x[1] for x in self.joint_limits])
        min_limits = np.tile(min_limits, [n_particles, 1])
        max_limits = np.tile(max_limits, [n_particles, 1])

        start_time = time.time()
        for k in range(n_iter):
            if time.time() - start_time > time_limit:
                break

            # update positions of particles
            particles_q += particles_v

            # apply joint limits
            min_viol = particles_q < min_limits
            max_viol = particles_q > max_limits
            particles_q[min_viol] = min_limits[min_viol]
            particles_q[max_viol] = max_limits[max_viol]

            # update the costs
            costs = np.array(map(cost_fn, particles_q))

            # update the 'bests'
            gbest[gbestcost > costs] = particles_q[gbestcost > costs]
            gbestcost[gbestcost > costs] = costs[gbestcost > costs]

            idx = np.argmin(gbestcost)
            pbest = gbest[idx]
            pbestcost = gbestcost[idx]

            # update the velocity
            phi1 = 1#np.random.rand()
            phi2 = 1#np.random.rand()
            w=0.25
            c1=0.5
            c2=0.25
            particles_v = w*particles_v + c1*phi1*(pbest - particles_q) + c2*phi2*(gbest - particles_q)

            error = np.linalg.norm(self.endpoint_pos(pbest) - target_pos)
            if error < eps:
                break
            
        end_time = time.time()
        if verbose: print "Runtime = %g, error = %g, n_iter=%d" % (end_time-start_time, error, k)

        return pbest        

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
            limit_hit = []
            for angle, (lim_min, lim_max) in izip(joint_angles, self.joint_limits):
                limit_hit.append(angle < lim_min or angle > lim_max)
                angle = max(lim_min, angle)
                angle = min(angle, lim_max)
                angles.append(angle)

            return np.array(angles), np.array(limit_hit)

    @property 
    def n_joints(self):
        return len(self.link_lengths)

class PlanarXZKinematicChain2Link(PlanarXZKinematicChain):
    def __init__(self, link_lengths, *args, **kwargs):
        if not len(link_lengths) == 2:
            raise ValueError("Can't instantiate a 2-link arm with > 2 links!")

        super(PlanarXZKinematicChain2Link, self).__init__(link_lengths, *args, **kwargs)

    def inverse_kinematics(self, target_pos, q_start, **kwargs):
        return inv_kin_2D(target_pos, self.link_lengths[0], self.link_lengths[1])


def inv_kin_2D(pos, l_upperarm, l_forearm, vel=None):
    '''
    NOTE: This function is almost exactly the same as riglib.stereo_opengl.ik.inv_kin_2D.
    There can only be room for one...
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
    return np.array([-sh_pabd, -el_pflex])


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
