'''Docstring.'''

import numpy as np
import socket

from riglib import ismore, blackrock, source
from riglib.bmi.state_space_models import StateSpaceArmAssist, StateSpaceReHand, StateSpaceIsMore
from riglib.ismore import settings
from utils.constants import *

try:
    import armassist
    import rehand
except:
    import warnings
    warnings.warn('clone the iBMI repo and put it on the path!')

class ArmAssistPlant(object):
    '''Sends velocity commands and receives feedback over UDP. Can be used
    with either the real or simulated ArmAssist.
    '''

    def __init__(self, print_commands=False):
        self.print_commands = print_commands

        self.source = source.DataSource(ismore.ArmAssistData, bufferlen=5, name='armassist')
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
        print "setting ArmAssist speeds to 0"
        self.send_vel(np.zeros(3))
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

    def __init__(self, print_commands=False):
        self.print_commands = print_commands

        self.source = source.DataSource(ismore.ReHandData, bufferlen=5, name='rehand')
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
        print "setting ReHand speeds to 0"
        self.send_vel(np.zeros(4))
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

    def __init__(self, print_commands=False):
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


################################################        


class ArmAssistPlantNoUDP(object):
    '''Similar methods as ArmAssistPlant, but: 
        1) doesn't send/receive anything over UDP, and 
        2) uses simulated ArmAssist (can't be used with real ArmAssist).
       Use this plant to simulate having (near) instantaneous feedback.
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

    def init(self):
        pass

    def start(self):
        '''Start the ArmAssist simulation processes.'''
        
        self.aa.start()

    def stop(self):
        pass

    def send_vel(self, vel):
        vel = vel.copy()
        
        # units of vel should be: (cm/s, cm/s, rad/s)
        assert len(vel) == 3

        # don't need to convert from rad/s to deg/s
        # (aa_pic expects units of rad/s)

        vel = np.mat(vel).T
        self.aa.update_reference(vel)

    # make note -- no conversion needed

    def get_pos(self):
        return np.array(self.aa.get_state()['wf']).reshape((3,))

    def get_vel(self):
        return np.array(self.aa.get_state()['wf_dot']).reshape((3,))

    # a magic function that instantaneously moves the simulated ArmAssist to a 
    #   new position+orientation
    def set_pos(self, pos):
        '''Magically set position+orientation in units of (cm, cm, rad).'''
        wf = np.mat(pos).T
        self.aa._set_wf(wf)


class ReHandPlantNoUDP(object):
    '''Similar methods as ReHandPlant, but: 
        1) doesn't send/receive anything over UDP, and 
        2) uses simulated ReHand (can't be used with real ReHand).
       Use this plant to simulate having (near) instantaneous feedback.
    '''
    
    def __init__(self):
        # create ReHand process
        self.rh = rehand.ReHand(tstep=0.005)
        self.rh.daemon = True

    def init(self):
        pass

    def start(self):
        '''Start the ReHand simulation process.'''

        self.rh.start()

    def stop(self):
        pass

    def send_vel(self, vel):
        vel = vel.copy()

        # units of vel should be: (rad/s, rad/s, rad/s, rad/s)
        assert len(vel) == 4
        
        # don't need to convert from rad/s to deg/s
        # (rh expects units of rad/s)

        vel = np.mat(vel).T
        self.rh.set_vel(vel)

    # no conversion needed (everything already in units of rad)

    def get_pos(self):
        return np.array(self.rh.get_state()['pos']).reshape((4,))

    def get_vel(self):
        return np.array(self.rh.get_state()['vel']).reshape((4,))

    # a magic function that instantaneously sets the simulated ReHand's angles
    def set_pos(self, pos):
        '''Magically set angles in units of (rad, rad, rad, rad).'''
        self.rh._set_pos(pos)


class IsMorePlantNoUDP(object):
    '''Similar methods as IsMorePlant, but: 
        1) doesn't send/receive anything over UDP, and 
        2) uses simulated ArmAssist+ReHand (can't be used with real devices).
       Use this plant to simulate having (near) instantaneous feedback.
    '''

    def __init__(self):
        self.aa_plant = ArmAssistPlantNoUDP()
        self.rh_plant = ReHandPlantNoUDP()

    def init(self):
        pass

    def start(self):
        '''Start the ArmAssist and ReHand simulation processes.'''

        self.aa_plant.start()
        self.rh_plant.start()

    def stop(self):
        pass

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

    # a magic function that instantaneously moves the simulated ArmAssist to a 
    #   new position+orientation and sets the simulated ReHand's angles
    def set_pos(self, pos):
        '''Magically set ArmAssist's position+orientation in units of 
        (cm, cm, rad) and ReHand's angles in units of (rad, rad, rad, rad).
        '''
        self.aa_plant.set_pos(pos[0:3])
        self.aa_plant.set_pos(pos[3:7])

