#!/usr/bin/python
'''
Representations of plants (control systems)
'''
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import numpy as np
from .stereo_opengl.primitives import Cylinder, Sphere, Cone, Cube, Chain
from .bmi import robot_arms

import math
import time
import socket
import select
import numpy as np
from collections import namedtuple

from utils.constants import *

class RefTrajectories(dict):
    '''
    Generic class to hold trajectories to be replayed by a plant.
    For now, this class is just a dictionary that has had its type changed
    '''
    pass


from riglib.source import DataSourceSystem
class FeedbackData(DataSourceSystem):
    '''
    Generic class for parsing UDP feedback data from a plant. Meant to be used with
    riglib.source.DataSource to grab and log data asynchronously.

    See DataSourceSystem for notes on the source interface
    '''

    MAX_MSG_LEN = 300
    sleep_time = 0

    # must define these in subclasses
    update_freq = None
    address     = None
    dtype       = None

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.address)

        # self.file_ = open(self.feedback_filename, 'w')

    def start(self):
        self.listening = True
        self.data = self.get_feedback_data()

    def stop(self):
        self.listening = False
        self.sock.close()
        # self.file_.close()

    def __del__(self):
        # The stop commands for the socket should be issued before this object is garbage-collected, but just in case...
        self.stop()

    def get(self):
        return next(self.data)

    def get_feedback_data(self):
        '''Yield received feedback data.'''

        self.last_timestamp = -1

        while self.listening:
            r, _, _ = select.select([self.sock], [], [], 0)

            if r:  # if the list r is not empty
                feedback = self.sock.recv(self.MAX_MSG_LEN)
                ts_arrival = time.time()  # secs

                # print "feedback:", feedback
                # self.file_.write(feedback.rstrip('\r') + "\n")

                processed_feedback = self.process_received_feedback(feedback, ts_arrival)

                if processed_feedback['ts'] != self.last_timestamp:
                    yield processed_feedback

                self.last_timestamp = processed_feedback['ts']

            time.sleep(self.sleep_time)

    def process_received_feedback(self, feedback, ts_arrival):
        raise NotImplementedError('Implement in subclasses!')




class Plant(object):
    '''
    Generic interface for task-plant interaction
    '''
    hdf_attrs = []
    def __init__(self, *args, **kwargs):
        pass

    def drive(self, decoder):
        '''
        Call this function to 'drive' the plant to the state specified by the decoder

        Parameters
        ----------
        decoder : bmi.Decoder instance
            Decoder used to estimate the state of/control the plant

        Returns
        -------
        None
        '''
        # Instruct the plant to go to the decoder-specified intrinsic coordinates
        # decoder['q'] is a special __getitem__ case. See riglib.bmi.Decoder.__getitem__/__setitem__
        self.set_intrinsic_coordinates(decoder['q'])

        # Not all intrinsic coordinates will be achievable. So determine where the plant actually went
        intrinsic_coords = self.get_intrinsic_coordinates()

        # Update the decoder state with the current state of the plant, after the last command
        if not np.any(np.isnan(intrinsic_coords)):
            decoder['q'] = self.get_intrinsic_coordinates()

    def get_data_to_save(self):
        '''
        Get data to save regarding the state of the plant on every iteration of the event loop

        Parameters
        ----------
        None

        Returns
        -------
        dict:
            keys are strings, values are np.ndarray objects of data values
        '''
        return dict()

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

    def init_decoder(self, decoder):
        decoder['q'] = self.get_intrinsic_coordinates()


class AsynchronousPlant(Plant):
    def init(self):
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.source)
        super(AsynchronousPlant, self).init()

    def start(self):
        '''
        Only start the DataSource after it has been registered with
        the sink manager in the call to init()
        '''
        self.source.start()
        super(AsynchronousPlant, self).start()

    def stop(self):
        self.source.stop()
        super(AsynchronousPlant, self).stop()

###################################################
##### Virtual plants for specific experiments #####
###################################################
class CursorPlant(Plant):
    '''
    Create a plant which is a 2-D or 3-D cursor on a screen/stereo display
    '''
    hdf_attrs = [('cursor', 'f8', (3,))]
    def __init__(self, endpt_bounds=None, cursor_radius=0.4, cursor_color=(.5, 0, .5, 1), starting_pos=np.array([0., 0., 0.]), vel_wall=True, **kwargs):
        self.endpt_bounds = endpt_bounds
        self.position = starting_pos
        self.cursor_radius = cursor_radius
        self.cursor_color = cursor_color
        self._pickle_init()
        self.vel_wall = vel_wall

    def _pickle_init(self):
        self.cursor = Sphere(radius=self.cursor_radius, color=self.cursor_color)
        self.cursor.translate(*self.position, reset=True)
        self.graphics_models = [self.cursor]

    def draw(self):
        self.cursor.translate(*self.position, reset=True)

    def set_color(self, color):
        self.cursor_color = color
        self.cursor.color = color

    def set_bounds(self, bounds):
        self.endpt_bounds = bounds

    def set_cursor_radius(self, radius):
        self.cursor_radius = radius
        self.cursor = Sphere(radius=radius, color=self.cursor_color)
        self.graphics_models = [self.cursor]

    def get_endpoint_pos(self):
        return self.position

    def set_endpoint_pos(self, pt, **kwargs):
        self.set_intrinsic_coordinates(pt)
        self.draw()

    def get_intrinsic_coordinates(self):
        return self.position

    def set_intrinsic_coordinates(self, pt):
        self.position = self._bound(pt, [])[0]
        self.draw()

    def set_visibility(self, visible):
        self.visible = visible
        if visible:
            self.graphics_models[0].attach()
        else:
            self.graphics_models[0].detach()

    def _bound(self, pos, vel):
        pos = pos.copy()
        vel = vel.copy()
        if len(vel) == 0:
            vel_wall = self.vel_wall # don't worry about vel if it's empty
            self.vel_wall = False
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
        if len(vel) == 0:
            self.vel_wall = vel_wall # restore previous value
        return pos, vel

    def drive(self, decoder):
        pos = decoder['q'].copy()
        vel = decoder['qdot'].copy()

        pos, vel = self._bound(pos, vel)

        decoder['q'] = pos
        decoder['qdot'] = vel
        super(CursorPlant, self).drive(decoder)

    def get_data_to_save(self):
        return dict(cursor=self.position)


class AuditoryCursor(Plant):
    '''
    An auditory cursor that changes frequency accordingly
    '''
    hdf_attrs = [('aud_cursor_freq', 'f8', (1,))]

    def __init__(self, min_freq, max_freq, sound_duration=0.1):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.bits = 16

        import pygame
        pygame.mixer.pre_init(44100, -self.bits, 2)
        pygame.init()

        duration = 0.1  # in seconds
        self.sample_rate = 44100
        self.n_samples = int(round(duration*self.sample_rate))
        self.max_sample = 2**(self.bits - 1) - 1
        self.freq = 0
        #setup our numpy array to handle 16 bit ints, which is what we set our mixer to expect with "bits" up above
        self.buf = np.zeros((self.n_samples, 2), dtype = np.int16)
        self.buf_ext = np.zeros((10*self.n_samples, 2), dtype=np.int16)
        self.t = np.arange(self.n_samples)/float(self.n_samples)*duration
        self.t0 = np.arange(self.n_samples)*0
        self.t_start = time.time()

    def drive(self, decoder):
        self.freq = decoder.filt.F

        if np.logical_and(decoder.cnt == 0, decoder.feedback):
            #Just got reset:
            if self.freq > self.max_freq:
                self.freq = self.max_freq
            elif self.freq < self.min_freq:
                self.freq = self.min_freq
            self.play_freq()

    def play_freq(self):
        self.buf[:,0] = np.round(self.max_sample*np.sin(2*math.pi*self.freq*self.t)).astype(int)
        self.buf[:,1] = np.round(self.max_sample*np.sin(2*math.pi*self.freq*self.t0)).astype(int)
        import pygame
        sound = pygame.sndarray.make_sound(self.buf)
        sound.play()

    def play_white_noise(self):
        self.buf_ext[:,0] = np.round(self.max_sample*np.random.normal(0, self.max_sample/2., (10*self.n_samples, ))).astype(int)
        self.buf_ext[:,1] = np.round(self.max_sample*np.zeros((10*self.n_samples, ))).astype(int)
        import pygame
        sound = pygame.sndarray.make_sound(self.buf_ext)
        sound.play()


    def get_intrinsic_coordinates(self):
        return self.freq


class onedimLFP_CursorPlant(CursorPlant):
    '''
    A square cursor confined to vertical movement
    '''
    hdf_attrs = [('lfp_cursor', 'f8', (3,))]

    def __init__(self, endpt_bounds, *args, **kwargs):
        self.lfp_cursor_rad = kwargs['lfp_cursor_rad']
        self.lfp_cursor_color = kwargs['lfp_cursor_color']
        args=[(), kwargs['lfp_cursor_color']]
        super(onedimLFP_CursorPlant, self).__init__(endpt_bounds, *args, **kwargs)


    def _pickle_init(self):
        self.cursor = Cube(side_len=self.lfp_cursor_rad, color=self.lfp_cursor_color)
        self.cursor.translate(*self.position, reset=True)
        self.graphics_models = [self.cursor]

    def drive(self, decoder):
        pos = decoder.filt.get_mean()
        pos = [-8, -2.2, pos]

        if self.endpt_bounds is not None:
            if pos[2] < self.endpt_bounds[4]:
                pos[2] = self.endpt_bounds[4]

            if pos[2] > self.endpt_bounds[5]:
                pos[2] = self.endpt_bounds[5]

            self.position = pos
            self.draw()

    def turn_off(self):
        self.cursor.detach()

    def turn_on(self):
        self.cursor.attach()

    def get_data_to_save(self):
        return dict(lfp_cursor=self.position)

class onedimLFP_CursorPlant_inverted(onedimLFP_CursorPlant):
    '''
    A square cursor confined to vertical movement
    '''
    hdf_attrs = [('lfp_cursor', 'f8', (3,))]

    def drive(self, decoder):
        std_pos = decoder.filt.get_mean()
        inv_pos = [-8, -2.2, -1.0*std_pos]

        if self.endpt_bounds is not None:
            if inv_pos[2] < self.endpt_bounds[4]:
                inv_pos[2] = self.endpt_bounds[4]

            if inv_pos[2] > self.endpt_bounds[5]:
                inv_pos[2] = self.endpt_bounds[5]

            self.position = inv_pos
            self.draw()

class twodimLFP_CursorPlant(onedimLFP_CursorPlant):
    '''Same as 1d cursor but assumes decoder returns array '''
    def drive(self, decoder):
        #Pos = (Left-Right, 0, Up-Down)
        pos = decoder.filt.get_mean()
        pos = [pos[0], -2.2, pos[2]]
        #pos = [-8, -2.2, pos[2]]

        if self.endpt_bounds is not None:
            if pos[2] < self.endpt_bounds[4]:
                pos[2] = self.endpt_bounds[4]

            if pos[2] > self.endpt_bounds[5]:
                pos[2] = self.endpt_bounds[5]

            self.position = pos
            self.draw()


arm_color = (181/256., 116/256., 96/256., 1)
arm_radius = 0.6
pi = np.pi
class RobotArmGen2D(Plant):
    '''
    Generic virtual plant for creating a kinematic chain of any number of links but confined to the X-Z (vertical) plane
    '''
    def __init__(self, link_radii=arm_radius, joint_radii=arm_radius, link_lengths=[15,15,5,5], joint_colors=arm_color,
        link_colors=arm_color, base_loc=np.array([2., 0., -15]), joint_limits=[(-pi,pi), (-pi,0), (-pi/2,pi/2), (-pi/2, 10*pi/180)], stay_on_screen=False, **kwargs):
        '''
        Instantiate the graphics and the virtual arm for a planar kinematic chain
        '''
        self.num_joints = num_joints = len(link_lengths)

        self.link_lengths = link_lengths
        self.curr_vecs = np.zeros([num_joints, 3]) #rows go from proximal to distal links

        # set initial vecs to correct orientations (arm starts out vertical)
        self.curr_vecs[0,2] = self.link_lengths[0]
        self.curr_vecs[1:,0] = self.link_lengths[1:]

        # Instantiate the kinematic chain object
        self.kin_chain = self.kin_chain_class(link_lengths, base_loc=base_loc)
        self.kin_chain.joint_limits = joint_limits

        self.base_loc = base_loc

        self.link_colors = link_colors
        self.chain = Chain(link_radii, joint_radii, link_lengths, joint_colors, link_colors)
        self.cursor = Sphere(radius=arm_radius/2, color=link_colors)
        self.graphics_models = [self.chain.link_groups[0], self.cursor]

        self.chain.translate(*self.base_loc, reset=True)

        self.hdf_attrs = [('cursor', 'f8', (3,)), ('joint_angles','f8', (self.num_joints, )), ('arm_visible', 'f8', (1,))]

        self.visible = True # arm is visible when initialized

        self.stay_on_screen = stay_on_screen
        self.joint_angles = np.zeros(self.num_joints)

    @property
    def kin_chain_class(self):
        return robot_arms.PlanarXZKinematicChain

    def set_color(self, color):
        self.link_colors = color
        self.cursor.color = color

    def set_bounds(self, bounds):
        '''
        For compatibility with other cursor plants
        '''
        pass 

    def set_cursor_radius(self, radius):
        self.cursor_radius = radius
        del self.cursor
        self.cursor = Sphere(radius=radius, color=self.link_colors)
        self.graphics_models[1] = self.cursor

    def get_endpoint_pos(self):
        '''
        Returns the current position of the non-anchored end of the arm.
        '''
        return self.kin_chain.endpoint_pos(self.joint_angles)

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
        angles = self.kin_chain.inverse_kinematics(pos, q_start=self.get_intrinsic_coordinates(), verbose=False, eps=0.008, **kwargs).ravel()
        return angles

    def calc_joint_angles(self, vecs):
        return np.arctan2(vecs[:,2], vecs[:,0])

    def get_intrinsic_coordinates(self):
        '''
        Returns the joint angles of the arm in radians
        '''
        return self.calc_joint_angles(self.curr_vecs)

    def set_intrinsic_coordinates(self, theta):
        '''
        Set the joints by specifying the angles in radians.

        Parameters
        ----------
        theta : np.ndarray
            Theta is a list of angles. If an element of theta = NaN, angle should remain the same.

        Returns
        -------
        None
        '''
        new_endpt_pos = self.kin_chain.endpoint_pos(theta)
        if self.stay_on_screen and (new_endpt_pos[0] > 25 or new_endpt_pos[0] < -25 or new_endpt_pos[-1] < -14 or new_endpt_pos[-1] > 14):
            # ignore the command because it would push the endpoint off the screen
            return

        if not np.any(np.isnan(theta)):
            self.joint_angles = theta
            for i in range(self.num_joints):
                self.curr_vecs[i] = self.link_lengths[i]*np.array([np.cos(theta[i]), 0, np.sin(theta[i])])

        self.chain._update_link_graphics(self.curr_vecs)
        self.cursor.translate(*self.get_endpoint_pos(), reset=True)

    def get_data_to_save(self):
        return dict(cursor=self.get_endpoint_pos(), joint_angles=self.get_intrinsic_coordinates(), arm_visible=self.visible)

    def set_visibility(self, visible):
        self.visible = visible
        if visible:
            self.graphics_models[0].attach()
        else:
            self.graphics_models[0].detach()

class EndptControlled2LArm(RobotArmGen2D):
    '''
    2-link arm controlled in extrinsic coordinates (endpoint position)
    '''
    def __init__(self, *args, **kwargs):
        super(EndptControlled2LArm, self).__init__(*args, **kwargs)
        self.hdf_attrs = [('cursor', 'f8', (3,)), ('arm_visible','f8', (1,))]

    def get_intrinsic_coordinates(self):
        return self.get_endpoint_pos()

    def set_intrinsic_coordinates(self, pos, **kwargs):
        self.set_endpoint_pos(pos, **kwargs)

    def set_endpoint_pos(self, pos, **kwargs):
        if pos is not None:
            # Run the inverse kinematics
            theta = self.perform_ik(pos, **kwargs)
            self.joint_angles = theta

            for i in range(self.num_joints):
                if theta[i] is not None and ~np.isnan(theta[i]):
                    self.curr_vecs[i] = self.link_lengths[i]*np.array([np.cos(theta[i]), 0, np.sin(theta[i])])

            self.chain._update_link_graphics(self.curr_vecs)

    def get_data_to_save(self):
        return dict(cursor=self.get_endpoint_pos(), arm_visible=self.visible)

    @property
    def kin_chain_class(self):
        return robot_arms.PlanarXZKinematicChain2Link
