'''See the shared Google Drive documentation for an inheritance diagram that
shows the relationships between the classes defined in this file.
'''

import numpy as np
import socket
import time

from riglib import source
from riglib.ismore import settings, udp_feedback_client
from tasks import ismore_bmi_lib
from utils.constants import *

import armassist
import rehand


class BasePlant(object):
    def __init__(self):
        raise NotImplementedError('Implement in subclasses!')

    def init(self):
        raise NotImplementedError('Implement in subclasses!')

    def start(self):
        raise NotImplementedError('Implement in subclasses!')

    def stop(self):
        raise NotImplementedError('Implement in subclasses!')

    def last_data_ts_arrival(self):
        raise NotImplementedError('Implement in subclasses!')

    def send_vel(self, vel):
        raise NotImplementedError('Implement in subclasses!')

    def get_pos(self):
        raise NotImplementedError('Implement in subclasses!')

    def get_vel(self):
        raise NotImplementedError('Implement in subclasses!')

    def enable(self):
        '''Disable the device's motor drivers.'''
        raise NotImplementedError('Implement in subclasses!')

    def disable(self):
        '''Disable the device's motor drivers.'''
        raise NotImplementedError('Implement in subclasses!')

    def enable_watchdog(self, timeout_ms):
        raise NotImplementedError('Implement in subclasses!')


class BasePlantUDP(BasePlant):
    '''Abstract base class.'''

    # define in subclasses!
    ssm_cls           = None
    addr              = None
    feedback_data_cls = None
    data_source_name  = None
    n_dof             = None

    def __init__(self):
        self.source = source.DataSource(self.feedback_data_cls, bufferlen=5, name=self.data_source_name)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # used only for sending

        ssm = self.ssm_cls()
        self.pos_state_names = [s.name for s in ssm.states if s.order == 0]
        self.vel_state_names = [s.name for s in ssm.states if s.order == 1]

    def init(self):
        from riglib import sink
        sink.sinks.register(self.source)

    def start(self):
        # only start this DataSource after it has been registered with 
        # the SinkManager singleton (sink.sinks) in the call to init()
        self.source.start()
        self.ts_start_data = time.time()

    def stop(self):
        self.send_vel(np.zeros(self.n_dof))
        self.source.stop()

    def last_data_ts_arrival(self):
        return self.source.read(n_pts=1)['ts_arrival'][0]

    def _send_command(self, command):
        self.sock.sendto(command, self.addr)


class ArmAssistPlantUDP(BasePlantUDP):
    '''Sends velocity commands and receives feedback over UDP. Can be used
    with either the real or simulated ArmAssist.
    '''

    ssm_cls           = ismore_bmi_lib.StateSpaceArmAssist
    addr              = settings.ARMASSIST_UDP_SERVER_ADDR
    feedback_data_cls = udp_feedback_client.ArmAssistData
    data_source_name  = 'armassist'
    n_dof             = 3

    def send_vel(self, vel):
        vel = vel.copy()

        # units of vel should be: [cm/s, cm/s, rad/s]
        assert len(vel) == 3

        # convert units to: [mm/s, mm/s, deg/s]
        vel[0] *= cm_to_mm
        vel[1] *= cm_to_mm
        vel[2] *= rad_to_deg

        self._send_command('SetSpeed ArmAssist %f %f %f\r' % tuple(vel))

    def get_pos(self):
        return np.array(tuple(self.source.read(n_pts=1)['data'][self.pos_state_names][0]))     

    def get_vel(self):
        pos = self.source.read(n_pts=2)['data'][self.pos_state_names]
        ts = self.source.read(n_pts=2)['ts']

        delta_pos = np.array(tuple(pos[1])) - np.array(tuple(pos[0]))
        delta_ts  = ts[1] - ts[0]
        
        vel = delta_pos / delta_ts

        if ts[0] != 0 and any(np.isnan(v) for v in vel):
            print "WARNING -- delta_ts = 0 in AA vel calculation:", vel
            for i in range(3):
                if np.isnan(vel[i]):
                    vel[i] = 0

        return vel

    def enable(self):
        self._send_command('SetControlMode ArmAssist Global\r')

    def disable(self):
        self._send_command('SetControlMode ArmAssist Disable\r')

    def enable_watchdog(self, timeout_ms):
        print 'ArmAssist watchdog not enabled, doing nothing'


class ReHandPlantUDP(BasePlantUDP):
    '''Sends velocity commands and receives feedback over UDP. Can be used
    with either the real or simulated ReHand.
    '''

    ssm_cls           = ismore_bmi_lib.StateSpaceReHand
    addr              = settings.REHAND_UDP_SERVER_ADDR
    feedback_data_cls = udp_feedback_client.ReHandData
    data_source_name  = 'rehand'
    n_dof             = 4

    def send_vel(self, vel):
        vel = vel.copy()

        # units of vel should be: [rad/s, rad/s, rad/s, rad/s]
        assert len(vel) == 4
        
        # convert units to: [deg/s, deg/s, deg/s, deg/s]
        vel *= rad_to_deg

        self._send_command('SetSpeed ReHand %f %f %f %f\r' % tuple(vel))

    def get_pos(self):
        return np.array(tuple(self.source.read(n_pts=1)['data'][self.pos_state_names][0]))

    def get_vel(self):
        return np.array(tuple(self.source.read(n_pts=1)['data'][self.vel_state_names][0]))

    def enable(self):
        self._send_command('SystemEnable ReHand\r')

    def disable(self):
        self._send_command('SystemDisable ReHand\r')

    def enable_watchdog(self, timeout_ms):
        self._send_command('WatchDogEnable ReHand %d\r' % timeout_ms)


################################################        


class BasePlantNonUDP(BasePlant):
    
    def init(self):
        pass

    def stop(self):
        pass

    def enable(self):
        pass

    def last_data_ts_arrival(self):
        # there's no delay when receiving feedback using the NonUDP classes, 
        #   since nothing is being sent over UDP and feedback data can be 
        #   requested at any time 
        return time.time()

    def disable(self):
        pass

    def enable_watchdog(self, timeout_ms):
        pass


class ArmAssistPlantNonUDP(BasePlantNonUDP):
    '''Similar methods as ArmAssistPlantUDP, but: 
        1) doesn't send/receive anything over UDP, and 
        2) uses simulated ArmAssist (can't be used with real ArmAssist).
       Use this plant to simulate having (near) instantaneous feedback.
    '''

    def __init__(self):
        # create ArmAssist process
        aa_tstep = 0.005                  # how often the simulated ArmAssist moves itself
        aa_pic_tstep = 0.01               # how often the simulated ArmAssist PI controller acts
        KP = np.mat([[-10.,   0.,  0.],
                     [  0., -20.,  0.],
                     [  0.,   0., 20.]])  # P gain matrix
        TI = 0.1 * np.identity(3)         # I gain matrix

        self.aa = armassist.ArmAssist(aa_tstep, aa_pic_tstep, KP, TI)
        self.aa.daemon = True

    def start(self):
        '''Start the ArmAssist simulation processes.'''
        
        self.aa.start()
        self.ts_start_data = time.time()

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


class ReHandPlantNonUDP(BasePlantNonUDP):
    '''Similar methods as ReHandPlantUDP, but: 
        1) doesn't send/receive anything over UDP, and 
        2) uses simulated ReHand (can't be used with real ReHand).
       Use this plant to simulate having (near) instantaneous feedback.
    '''
    
    def __init__(self):
        # create ReHand process
        self.rh = rehand.ReHand(tstep=0.005)
        self.rh.daemon = True

    def start(self):
        '''Start the ReHand simulation process.'''

        self.rh.start()
        self.ts_start_data = time.time()

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


################################################ 


class BasePlantIsMore(BasePlant):

    # define in subclasses!
    aa_plant_cls = None
    rh_plant_cls = None

    def __init__(self):
        self.aa_plant = self.aa_plant_cls()
        self.rh_plant = self.rh_plant_cls()

    def init(self):
        self.aa_plant.init()
        self.rh_plant.init()

    def start(self):
        self.aa_plant.start()
        self.rh_plant.start()
        self.ts_start_data = time.time()

    def stop(self):
        self.aa_plant.stop()
        self.rh_plant.stop()

    def last_data_ts_arrival(self):
        return {
            'ArmAssist': self.aa_plant.last_data_ts_arrival(), 
            'ReHand':    self.rh_plant.last_data_ts_arrival(),
        }

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

    def enable(self):
        self.aa_plant.enable()
        self.rh_plant.enable()

    def disable(self):
        self.aa_plant.disable()
        self.rh_plant.disable()


class IsMorePlantUDP(BasePlantIsMore):
    '''Sends velocity commands and receives feedback over UDP. Can be used
    with either the real or simulated ArmAssist+ReHand.
    '''

    aa_plant_cls = ArmAssistPlantUDP
    rh_plant_cls = ReHandPlantUDP


class IsMorePlantNonUDP(BasePlantIsMore):
    '''Similar methods as IsMorePlant, but: 
        1) doesn't send/receive anything over UDP, and 
        2) uses simulated ArmAssist+ReHand (can't be used with real devices).
       Use this plant to simulate having (near) instantaneous feedback.
    '''

    aa_plant_cls = ArmAssistPlantNonUDP
    rh_plant_cls = ReHandPlantNonUDP

    # a magic function that instantaneously moves the simulated ArmAssist to a 
    #   new position+orientation and sets the simulated ReHand's angles
    def set_pos(self, pos):
        '''Magically set ArmAssist's position+orientation in units of 
        (cm, cm, rad) and ReHand's angles in units of (rad, rad, rad, rad).
        '''
        self.aa_plant.set_pos(pos[0:3])
        self.rh_plant.set_pos(pos[3:7])


UDP_PLANT_CLS_DICT = {
    'ArmAssist': ArmAssistPlantUDP,
    'ReHand':    ReHandPlantUDP,
    'IsMore':    IsMorePlantUDP,
}

NONUDP_PLANT_CLS_DICT = {
    'ArmAssist': ArmAssistPlantNonUDP,
    'ReHand':    ReHandPlantNonUDP,
    'IsMore':    IsMorePlantNonUDP,
}
