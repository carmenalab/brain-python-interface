'''Docstring.'''

import numpy as np
import socket

from riglib import ismore, blackrock, source
from riglib.bmi.state_space_models import StateSpaceArmAssist, StateSpaceReHand, StateSpaceIsMore
from riglib.ismore import settings
from utils.constants import *

import armassist
import rehand

class ArmAssistPlant(object):
    '''Sends velocity commands and receives feedback over UDP. Can be used
    with either the real or simulated ArmAssist.
    '''

    def __init__(self, print_commands=True):
        self.print_commands = print_commands

        self.source = source.DataSource(ismore.ArmAssistData, name='armassist')  # TODO -- set small buffer length
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # used only for sending
        self.aa_addr = settings.armassist_udp_server
        
        command = 'SetControlMode ArmAssist Global\n'
        self.sock.sendto(command, self.aa_addr)
        
        ssm = StateSpaceArmAssist()
        self.pos_state_names = [s.name for s in ssm.states if s.order == 0]
        self.vel_state_names = [s.name for s in ssm.states if s.order == 1]

    def init(self):
        from riglib import sink
        sink.sinks.register(self.source)

    def start(self):
        # only start this DataSource after it has been registered with 
        # the SinkManager singleton (sink.sinks) in the call to init()
        self.source.start()

    def stop(self):
        self.source.stop()

    def send_vel(self, vel):
        vel = vel.copy()

        # units of vel should be: [cm/s, cm/s, rad/s]
        assert len(vel) == 3

        # convert units to: [mm/s, mm/s, deg/s]
        vel[0] *= cm_to_mm
        vel[1] *= cm_to_mm
        vel[2] *= rad_to_deg

        command = 'SetSpeed ArmAssist %f %f %f\r' % tuple(vel)
        self.sock.sendto(command, self.aa_addr)
        if self.print_commands:
            print 'sending command:', command

    def get_pos(self):
        return np.array(tuple(self.source.read(n_pts=1)['data'][self.pos_state_names][0]))     

    def get_vel(self):
        pos = self.source.read(n_pts=2)['data'][self.pos_state_names]
        ts = self.source.read(n_pts=2)['ts'][self.pos_state_names]

        delta_pos = np.array(tuple(pos[1])) - np.array(tuple(pos[0]))
        delta_ts  = np.array(tuple(ts[1])) - np.array(tuple(ts[0]))

        vel = delta_pos / (delta_ts * ms_to_s)

        if any(np.isnan(v) for v in vel):
            print "WARNING -- nans in vel:", vel
            print "pos", pos
            print "ts", ts
            print "delta_pos", delta_pos
            print "delta_ts", delta_ts
            for i in range(3):
                if np.isnan(vel[i]):
                    vel[i] = 0

        return vel


class ReHandPlant(object):
    '''Sends velocity commands and receives feedback over UDP. Can be used
    with either the real or simulated ReHand.
    '''

    def __init__(self, print_commands=True):
        self.print_commands = print_commands

        self.source = source.DataSource(ismore.ReHandData, name='rehand')     # TODO -- set small buffer length
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # used only for sending
        self.rh_addr = settings.rehand_udp_server

        ssm = StateSpaceReHand()
        self.pos_state_names = [s.name for s in ssm.states if s.order == 0]
        self.vel_state_names = [s.name for s in ssm.states if s.order == 1]

    def init(self):
        from riglib import sink
        sink.sinks.register(self.source)

    def start(self):
        # only start this DataSource after it has been registered with 
        # the SinkManager singleton (sink.sinks) in the call to init()
        self.source.start()

    def stop(self):
        self.source.stop()

    def send_vel(self, vel):
        vel = vel.copy()

        # units of vel should be: [rad/s, rad/s, rad/s, rad/s]
        assert len(vel) == 4
        
        # convert units to: [deg/s, deg/s, deg/s, deg/s]
        vel *= rad_to_deg

        command = 'SetSpeed ReHand %f %f %f %f\r' % tuple(vel)
        self.sock.sendto(command, self.rh_addr)
        if self.print_commands:
            print 'sending command:', command

    def get_pos(self):
        return np.array(tuple(self.source.read(n_pts=1)['data'][self.pos_state_names][0]))

    def get_vel(self):
        return np.array(tuple(self.source.read(n_pts=1)['data'][self.vel_state_names][0]))


class IsMorePlant(object):
    '''Sends velocity commands and receives feedback over UDP. Can be used
    with either the real or simulated ArmAssist+ReHand.
    '''

    def __init__(self, print_commands=True):
        self.aa_plant = ArmAssistPlant(print_commands=print_commands)
        self.rh_plant = ReHandPlant(print_commands=print_commands)

    def init(self):
        self.aa_plant.init()
        self.rh_plant.init()

    def start(self):
        self.aa_plant.start()
        self.rh_plant.start()

    def stop(self):
        self.aa_plant.stop()
        self.rh_plant.stop()

    def send_vel(self, vel):
        self.aa_plant.send_vel(vel[0:3])
        self.rh_plant.send_vel(vel[3:7])

    def get_pos(self):
        aa_pos = self.aa_plant.get_pos()
        rh_pos = self.rh_plant.get_pos()
        return np.hstack([aa_pos, rh_pos])

    def get_vel(self):
        aa_vel = self.aa_plant.get_vel()
        rh_vel = self.rh_plant.get_vel()
        return np.hstack([aa_vel, rh_vel])

        

class IsMorePlantNoUDP(object):
    '''Similar methods as IsMorePlant, but: 1) doesn't send/receive anything
    over UDP and 2) uses simulated ArmAssist and/or ReHand. Use this plant if
    you want to simulate having (near) instantaneous feedback.
    '''
    def __init__(self):
        # create and start ArmAssist object (includes ArmAssist and its PIC)
        aa_tstep = 0.005
        aa_pic_tstep = 0.01
        KP = np.mat([[-10.,   0.,  0.],
                     [  0., -20.,  0.],
                     [  0.,   0., 20.]])  # P gain matrix
        TI = 0.1*np.identity(3)  # I gain matrix

        self.aa = armassist.ArmAssist(aa_tstep, aa_pic_tstep, KP, TI)
        self.aa.daemon = True


        self.rh = rehand.ReHand(tstep=0.005)
        self.rh.daemon = True

    # # a "magic" function that instantaneously moves the ArmAssist and ReHand to a new configuration
    # # IMPORTANT: only use to set initial position/orientation
    # def set_pos(self, pos):
    #     '''Magically set position (x, y, psi) in units of (cm, cm, rad).'''
    #     wf = np.mat(pos).T
    #     self.aa._set_wf(wf)

    def init(self):
        pass

    def start(self):
        # start ArmAssist and ReHand simulation processes
        self.aa.start()
        self.rh.start()

    def stop(self):
        pass

    def send_vel(self, vel, dev):
        vel = vel.copy()
        if dev == 'ArmAssist':
            # units of vel should be: (cm/s, cm/s, rad/s)
            assert len(vel) == 3

            # don't need to convert from rad/s to deg/s
            # (aa_pic expects units of rad/s)

            vel = np.mat(vel).T
            self.aa.update_reference(vel)

        elif dev == 'ReHand':
            # units of vel should be: (rad/s, rad/s, rad/s, rad/s)
            assert len(vel) == 4
            
            # don't need to convert from rad/s to deg/s
            # (rh expects units of rad/s)

            vel = np.mat(vel).T
            self.rh.set_vel(vel)

        elif dev == 'IsMore':
            # units of vel should be: (cm/s, cm/s, rad/s, rad/s, rad/s, rad/s, rad/s)
            assert len(vel) == 7
            
            # don't need to convert from rad/s to deg/s
            # (aa_pic and rh expect units of rad/s)

            aa_vel = np.mat(vel[0:3]).T
            self.aa.update_reference(aa_vel)

            rh_vel = np.mat(vel[3:7]).T
            self.rh.set_vel(rh_vel)

        else:
            raise Exception('Unknown device: ' + str(dev))
        
    def _get_state(self):
        aa_state = self.aa.get_state()
        aa_pos = np.array(aa_state['wf']).reshape((3,))
        aa_vel = np.array(aa_state['wf_dot']).reshape((3,))

        rh_state = self.rh.get_state()
        rh_pos = np.array(rh_state['pos']).reshape((4,))
        rh_vel = np.array(rh_state['vel']).reshape((4,))

        # no conversion needed (everything already in units of rad)

        return aa_pos, aa_vel, rh_pos, rh_vel

    def get_pos(self, dev):
        aa_pos, _, rh_pos, _ = self._get_state()

        if dev == 'ArmAssist':
            return aa_pos
        elif dev == 'ReHand':
            return rh_pos
        elif dev == 'IsMore':
            return np.hstack([aa_pos, rh_pos])
        else:
            raise Exception('Unknown device: ' + str(dev))

    def get_vel(self, dev):
        _, aa_vel, _, rh_vel = self._get_state()

        if dev == 'ArmAssist':
            return aa_vel
        elif dev == 'ReHand':
            return rh_vel
        elif dev == 'IsMore':
            return np.hstack([aa_vel, rh_vel])
        else:
            raise Exception('Unknown device: ' + str(dev))
