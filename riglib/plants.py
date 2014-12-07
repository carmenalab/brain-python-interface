#!/usr/bin/python
'''
Representations of plants (control systems)
'''
import numpy as np
from stereo_opengl.primitives import Cylinder, Sphere, Cone
from stereo_opengl.models import Group
from riglib.bmi import robot_arms
from riglib.stereo_opengl.xfm import Quaternion

import sys
import time
import socket
import select
import numpy as np
from collections import namedtuple

from riglib.ismore import settings
from utils.constants import *
from riglib.bmi import state_space_models as ssm
from riglib.bmi import state_space_models
from riglib import source
import robot

field_names = ['data', 'ts', 'ts_sent', 'ts_arrival', 'freq']
ArmAssistFeedbackData = namedtuple("ArmAssistFeedbackData", field_names)
ReHandFeedbackData    = namedtuple("ReHandFeedbackData",    field_names)
PassiveExoFeedbackData = namedtuple("PassiveExoFeedbackData", ['data', 'ts_arrival'])


class Plant(object):
    '''
    Generic interface for task-plant interaction
    '''
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

    def init(self):
        '''
        Secondary initialization after object construction. Does nothing by default
        '''
        pass

    def start(self):
        '''
        Start any auxiliary processes used by the plant
        '''
        pass

    def stop(self):
        '''
        Stop any auxiliary processes used by the plant
        '''        
        pass


class FeedbackData(object):
    '''Abstract parent class, not meant to be instantiated.'''

    client_cls = None

    def __init__(self):
        self.client = self.client_cls()

    def start(self):
        self.client.start()
        self.data = self.client.get_feedback_data()

    def stop(self):
        self.client.stop()

    def get(self):
        d = self.data.next()
        return np.array([(tuple(d.data), tuple(d.ts), d.ts_sent, d.ts_arrival, d.freq)], dtype=self.dtype)

    @staticmethod
    def _get_dtype(state_names):
        sub_dtype_data = np.dtype([(name, np.float64) for name in state_names])
        sub_dtype_ts   = np.dtype([(name, np.int64)   for name in state_names])
        return np.dtype([('data',       sub_dtype_data),
                         ('ts',         sub_dtype_ts),
                         ('ts_sent',    np.float64),
                         ('ts_arrival', np.float64),
                         ('freq',       np.float64)])

class Client(object):
    '''
    Generic client for UDP data sources
    '''

    MAX_MSG_LEN = 200
    sleep_time = 0

    # TODO -- rename this function to something else?
    def _create_and_bind_socket(self):
        '''
        Create UDP receive socket and bind
        '''
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.address)

        self._init = True
        
    def start(self):
        self.listening = True

    def stop(self):
        self.listening = False
    
    def __del__(self):
        self.stop()

    def get_feedback_data(self):
        raise NotImplementedError('Implement in subclasses!')

import struct

class UpperArmPassiveExoClient(Client):
    ##  socket to read from
    address = ('10.0.0.1', 60000)
    send_port = ('10.0.0.12', 60001)

    #address = ('localhost', 60000) 
    #send_port = ('localhost', 60001)

    MAX_MSG_LEN = 48
    sleep_time = 1 #### In production mode, this should be 1./60

    def __init__(self):
        self._create_and_bind_socket()
        self.tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def get_feedback_data(self):
        '''
        Yield received feedback data.
        '''
        while self.listening:
            self.tx_sock.sendto('s', self.send_port)
            # Hang until the read socket is "ready" for reading
            # "r" represents the list of devices that are ready from the list that was passed in
            time.sleep(self.sleep_time)
            r, _, _ = select.select([self.sock], [], [], 0)
            
            if len(r) > 0:
                ts_arrival = int(time.time() * 1e6)
                
                data = self.sock.recvfrom(self.MAX_MSG_LEN)
                self.tx_sock.sendto('c', self.send_port)

                # unpack the data
                data = struct.unpack('>dddddd', data[0])

                yield PassiveExoFeedbackData(data=data, ts_arrival=ts_arrival)

class StateSpaceUpperArmPassiveExo(state_space_models.StateSpace):
    '''
    State space to represent the EFRI exoskeleton in passive mode
    '''
    def __init__(self):
        '''
        Constructor for StateSpaceUpperArmPassiveExo
        A 6-D state space is created to represent all the position states. No velocity states are included.
        '''
        super(StateSpaceUpperArmPassiveExo, self).__init__(
            state_space_models.State('q1', stochastic=False, drives_obs=False, min_val=-25., max_val=25., order=0),
            state_space_models.State('q2', stochastic=False, drives_obs=False, min_val=-10, max_val=10, order=0),
            state_space_models.State('q3', stochastic=False, drives_obs=False, min_val=-14., max_val=14., order=0),
            state_space_models.State('q4', stochastic=False, drives_obs=False, min_val=-14., max_val=14., order=0),
            state_space_models.State('q5', stochastic=False, drives_obs=False, min_val=-14., max_val=14., order=0),
            state_space_models.State('q6', stochastic=False, drives_obs=False, min_val=-14., max_val=14., order=0),
        )    

def _get_dtype2(state_names):
    sub_dtype_data = np.dtype([(name, np.float64) for name in state_names])
    sub_dtype_ts   = np.dtype([(name, np.int64)   for name in state_names])
    return np.dtype([('data',       sub_dtype_data),
                     ('ts_arrival', np.float64)])

class UpperArmPassiveExoData(FeedbackData):
    '''
    Docstring
    '''

    update_freq = 60.
    client_cls = UpperArmPassiveExoClient

    ssm = StateSpaceUpperArmPassiveExo()
    state_names = [s.name for s in StateSpaceUpperArmPassiveExo().states if s.order == 0]
    dtype = _get_dtype2(state_names)

    def get(self):
        d = self.data.next()
        return np.array([(tuple(d.data), d.ts_arrival)], dtype=self.dtype)        


class PassivePlant(Plant):
    def drive(self, *args, **kwargs):
        raise NotImplementedError("Passive plant cannot be driven!")


class AsynchronousPlant(Plant):
    def init(self):
        from riglib import sink
        sink.sinks.register(self.source)
        super(AsynchronousPlant, self).init()

    def start(self):
        '''
        Only start the DataSource after it has been registered with 
        The SinkManager singleton (sink.sinks) in the call to init()
        '''
        self.source.start()
        super(AsynchronousPlant, self).start()

    def stop(self):
        self.source.stop()    
        super(AsynchronousPlant, self).stop()




from riglib.bmi.robot_arms import KinematicChain

class PassiveExoChain(KinematicChain):
    def _init_serial_link(self):
        pi = np.pi
        
        d = np.array([-2.4767, -4.2709, -11.1398, 130, 7.0377, 152]) 
        # d4: length from the shoulder center to the elbow of the monkey; this is mechanically fix! unit is [mm]
        # d6: length from the elbow to the wrist; this is flexible depending on the actual value; unit is [mm]
        a = np.array([1.8654, -1.0149, 0.4966, -7.7437, -2.4387, 0])
        alpha = np.array([(-90-1.1344)*pi/180, pi/2, -pi/2, pi/2, -pi/2, 0])
        offsets = np.array([25.8054*pi/180, -95.1254*pi/180, 37.8311*pi/180, 11.9996*pi/180, -60.2*pi/180, 0.5283*pi/180])
        
        links = []
        for k in range(6):
            link = robot.Link(a=a[k], d=d[k], alpha=alpha[k], offset=offsets[k])
            links.append(link)
        
        r = robot.SerialLink(links)
        
        r.tool[0:3, -1] = np.array([0, 0, 45.]).reshape(-1,1)
        self.robot = r
        self.link_lengths = a



import struct
class UpperArmPassiveExo(PassivePlant):
    hdf_attrs = [('joint_angles', 'f8', (6,))]
    read_addr = ('10.0.0.1', 60000)
    def __init__(self, print_commands=True):
        self.print_commands = print_commands
        ssm = StateSpaceUpperArmPassiveExo()
        self.initialized = False
        self.rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ssm = StateSpaceUpperArmPassiveExo()
        self.tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.kin_chain = PassiveExoChain()

    def start(self):
        self.rx_sock.bind(('10.0.0.1', 60000))
        super(UpperArmPassiveExo, self).start()        

    def get_data_to_save(self):
        if self.initialized:
            data = self.rx_sock.recvfrom(48)
            self.tx_sock.sendto('c', ('10.0.0.12', 60001))
            joint_angles = np.array(struct.unpack('>dddddd', data[0]))
        else:
            joint_angles = np.array(np.ones(6)*np.nan)
            
        self.tx_sock.sendto('s', ('10.0.0.12', 60001))
        self.initialized = True

        # joint_angles = np.array(tuple(self.source.read(n_pts=1)['data'][0]))
        self.joint_angles = joint_angles
        return dict(joint_angles=joint_angles)

    def get_endpoint_pos(self):
        ### THIS IS A STUB FUNCTION UNTIL THE IK WORKS!
        return np.zeros(3)

    def stop(self):
        self.rx_sock.close()


class CursorPlant(Plant):
    hdf_attrs = [('cursor', 'f8', (3,))]
    def __init__(self, endpt_bounds=None, cursor_radius=0.4, cursor_color=(.5, 0, .5, 1), starting_pos=np.array([0., 0., 0.]), vel_wall=True, **kwargs):
        self.endpt_bounds = endpt_bounds
        self.position = starting_pos
        self.starting_pos = starting_pos
        self.cursor_radius = cursor_radius
        self.cursor_color = cursor_color
        from riglib.bmi import state_space_models
        self.ssm = state_space_models.StateSpaceEndptVel2D()
        self._pickle_init()
        self.vel_wall = vel_wall

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
                if self.vel_wall: vel[0] = 0
            if pos[0] > self.endpt_bounds[1]: 
                pos[0] = self.endpt_bounds[1]
                if self.vel_wall: vel[0] = 0

            if pos[1] < self.endpt_bounds[2]: 
                pos[1] = self.endpt_bounds[2]
                if self.vel_wall: vel[1] = 0
            if pos[1] > self.endpt_bounds[3]: 
                pos[1] = self.endpt_bounds[3]
                if self.vel_wall: vel[1] = 0

            if pos[2] < self.endpt_bounds[4]: 
                pos[2] = self.endpt_bounds[4]
                if self.vel_wall: vel[2] = 0
            if pos[2] > self.endpt_bounds[5]: 
                pos[2] = self.endpt_bounds[5]
                if self.vel_wall: vel[2] = 0
        
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

        self.visible = True # arm is visible when initialized

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
        
        self.cursor = Sphere(radius=self.link_radii[-1]/2, color=self.link_colors[-1])
        # self.cursor.translate(*self.get_endpoint_pos(), reset=True)
        self.graphics_models = [self.link_groups[0], self.cursor]        

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

        self.cursor.translate(*self.get_endpoint_pos(), reset=True)

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
        return dict(cursor=self.get_endpoint_pos(), joint_angles=self.get_intrinsic_coordinates(), arm_visible=self.visible)

    def set_visibility(self, visible):
        self.visible = visible
        if visible:
            self.graphics_models[0].attach()
        else:
            self.graphics_models[0].detach()