'''Docstring.'''

import numpy as np
import socket

from riglib import blackrock, source
from riglib.bmi.state_space_models import StateSpaceArmAssist, StateSpaceReHand, StateSpaceIsMore

import armassist
import rehand

# CONSTANTS
rad_to_deg = 180 / np.pi
deg_to_rad = np.pi / 180


class IsMorePlant(object):
    '''Sends velocity commands and receives feedback over UDP. Can be used
    with either the real or simulated ArmAssist and/or ReHand. Uses 2 separate
    data sources for ArmAssist and ReHand.
    '''
    def __init__(self, print_commands=True):
        self.print_commands = print_commands

        self.aa_source = source.DataSource(blackrock.ArmAssistData, name='armassist')  # TODO -- set small buffer length
        self.rh_source = source.DataSource(blackrock.ReHandData, name='rehand')     # TODO -- set small buffer length

        # used only for sending commands (not for receiving feedback)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.aa_addr = ('127.0.0.1', 5001)
        self.rh_addr = ('127.0.0.1', 5000)

        # TODO -- don't hardcode these lists here, use names from state space models instead
        ssm_armassist = StateSpaceArmAssist()
        ssm_rehand    = StateSpaceReHand()
        ssm_ismore    = StateSpaceIsMore()

        self.aa_p_state_names  = [s.name for s in ssm_armassist.states if s.order == 0]
        self.aa_v_state_names  = [s.name for s in ssm_armassist.states if s.order == 1]
        self.rh_p_state_names  = [s.name for s in ssm_rehand.states if s.order == 0]
        self.rh_v_state_names  = [s.name for s in ssm_rehand.states if s.order == 1]
        self.all_p_state_names = [s.name for s in ssm_ismore.states if s.order == 0]
        self.all_v_state_names = [s.name for s in ssm_ismore.states if s.order == 1]

    def init(self):
        from riglib import sink
        self.sinks = sink.sinks

        self.sinks.register(self.aa_source)
        self.sinks.register(self.rh_source)

    def start(self):
        # only want to start these DataSources after they have been
        # register with the SinkManager singleton (self.sinks)
        # in the call to init()
        self.aa_source.start()
        self.rh_source.start()

    def stop(self):
        self.aa_source.stop()
        self.rh_source.stop()

    def send_vel(self, vel, dev='IsMore'):
        vel = vel.copy()
        if dev == 'ArmAssist':
            # units of vel should be: (cm/s, cm/s, rad/s)
            assert len(vel) == 3

            # convert from rad/s to deg/s
            vel[2] *= rad_to_deg

            command = 'SetSpeed ArmAssist %f %f %f\r' % tuple(vel)
            self.sock.sendto(command, self.aa_addr)
            if self.print_commands:
                print 'sending command:', command
        
        elif dev == 'ReHand':
            # units of vel should be: (rad/s, rad/s, rad/s, rad/s)
            assert len(vel) == 4
            
            # convert from rad/s to deg/s
            vel *= rad_to_deg

            command = 'SetSpeed ReHand %f %f %f %f\r' % tuple(vel)
            self.sock.sendto(command, self.rh_addr)
            if self.print_commands:
                print 'sending command:', command
        
        elif dev == 'IsMore':
            # units of vel should be: (cm/s, cm/s, rad/s, rad/s, rad/s, rad/s, rad/s)
            assert len(vel) == 7
            
            # convert from rad/s to deg/s
            vel[2:] *= rad_to_deg

            command = 'SetSpeed ArmAssist %f %f %f\r' % tuple(vel[0:3])
            self.sock.sendto(command, self.aa_addr)
            if self.print_commands:
                print 'sending command:', command

            command = 'SetSpeed ReHand %f %f %f %f\r' % tuple(vel[3:7])
            self.sock.sendto(command, self.rh_addr)
            if self.print_commands:
                print 'sending command:', command
        
        else:
            raise Exception('Unknown device: ' + str(dev))

    # Note: for get_pos and get_vel, conversion from deg to rad occurs inside
    # udp_feedback_client.py

    def get_pos(self, dev='IsMore'):
        if dev == 'ArmAssist':
            feedback = np.array(tuple(self.aa_source.read(n_pts=1)[0]))
            pos = feedback[0:3]
        elif dev == 'ReHand':
            feedback = np.array(tuple(self.rh_source.read(n_pts=1)[0]))
            pos = feedback[0:4]
        elif dev == 'IsMore':
            feedback = np.array(tuple(self.aa_source.read(n_pts=1)[0]))
            aa_pos = feedback[0:3]

            feedback = np.array(tuple(self.rh_source.read(n_pts=1)[0]))
            rh_pos = feedback[0:4]

            pos = np.hstack([aa_pos, rh_pos]) 
        else:
            raise Exception('Unknown device: ' + str(dev))

        return pos

    def get_vel(self, dev='IsMore'):
        if dev == 'ArmAssist':
            feedback = np.array(tuple(self.aa_source.read(n_pts=1)[0]))
            vel = feedback[3:6]
        elif dev == 'ReHand':
            feedback = np.array(tuple(self.rh_source.read(n_pts=1)[0]))
            vel = feedback[4:8]
        elif dev == 'IsMore':
            feedback = np.array(tuple(self.aa_source.read(n_pts=1)[0]))
            aa_vel = feedback[3:6]

            feedback = np.array(tuple(self.rh_source.read(n_pts=1)[0]))
            rh_vel = feedback[4:8]

            vel = np.hstack([aa_vel, rh_vel]) 
        else:
            raise Exception('Unknown device: ' + str(dev))

        return vel

        

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

    def send_vel(self, vel, dev='IsMore'):
        vel = vel.copy()
        if dev == 'ArmAssist':
            # units of vel should be: (cm/s, cm/s, rad/s)
            assert len(vel) == 3

            # don't need to convert from rad/s to deg/s
            # (aa_pic expects units of rad/s)

            vel = np.mat(vel).T
            # self.aa_pic.update_reference(vel)
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
            # self.aa_pic.update_reference(aa_vel)
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

    def get_pos(self, dev='IsMore'):
        aa_pos, _, rh_pos, _ = self._get_state()

        if dev == 'ArmAssist':
            return aa_pos
        elif dev == 'ReHand':
            return rh_pos
        elif dev == 'IsMore':
            return np.hstack([aa_pos, rh_pos])
        else:
            raise Exception('Unknown device: ' + str(dev))

    def get_vel(self, dev='IsMore'):
        _, aa_vel, _, rh_vel = self._get_state()

        if dev == 'ArmAssist':
            return aa_vel
        elif dev == 'ReHand':
            return rh_vel
        elif dev == 'IsMore':
            return np.hstack([aa_vel, rh_vel])
        else:
            raise Exception('Unknown device: ' + str(dev))



# class IsMorePlantOld(object):
#     '''Sends velocity commands and receives feedback over UDP. Can be used
#     with either the real or simulated ArmAssist and/or ReHand. Uses a single
#     multi-chan data source for ArmAssist and ReHand.
#      '''
#     def __init__(self):
#         ismore_ss = StateSpaceIsMore()
#         channels = ismore_ss.state_names

#         self.feedback_source = source.MultiChanDataSource(blackrock.FeedbackData, channels=channels)

#         # used only for sending commands (not for receiving feedback)
#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#         self.aa_addr = ('127.0.0.1', 5001)
#         self.rh_addr = ('127.0.0.1', 5000)

#         # TODO -- don't hardcode these lists here, use names from state space models instead
#         self.aa_p_state_names = ['aa_px', 'aa_py', 'aa_ppsi']
#         self.aa_v_state_names = ['aa_vx', 'aa_vy', 'aa_vpsi']
#         self.rh_p_state_names = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']
#         self.rh_v_state_names = ['rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']
#         self.all_p_state_names = self.aa_p_state_names + self.rh_p_state_names
#         self.all_v_state_names = self.aa_v_state_names + self.rh_v_state_names

#     def init(self):
#         pass

#     def start(self):
#         self.feedback_source.start()

#     def stop(self):
#         self.feedback_source.stop()

#     def send_vel(self, vel, dev='IsMore'):
#         if dev == 'ArmAssist':
#             # units of vel should be: (cm/s, cm/s, rad/s)
#             assert len(vel) == 3

#             # convert from rad/s to deg/s
#             vel[2] *= rad_to_deg

#             command = 'SetSpeed ArmAssist %f %f %f\r' % tuple(vel)
#             self.sock.sendto(command, self.aa_addr)
#             print 'sending command:', command
        
#         elif dev == 'ReHand':
#             # units of vel should be: (rad/s, rad/s, rad/s, rad/s)
#             assert len(vel) == 4
            
#             # convert from rad/s to deg/s
#             vel *= rad_to_deg

#             command = 'SetSpeed ReHand %f %f %f %f\r' % tuple(vel)
#             self.sock.sendto(command, self.rh_addr)
#             print 'sending command:', command
        
#         elif dev == 'IsMore':
#             # units of vel should be: (cm/s, cm/s, rad/s, rad/s, rad/s, rad/s, rad/s)
#             assert len(vel) == 7
            
#             # convert from rad/s to deg/s
#             vel[2:] *= rad_to_deg

#             command = 'SetSpeed ArmAssist %f %f %f\r' % tuple(vel[0:3])
#             self.sock.sendto(command, self.aa_addr)
#             print 'sending command:', command

#             command = 'SetSpeed ReHand %f %f %f %f\r' % tuple(vel[3:7])
#             self.sock.sendto(command, self.rh_addr)
#             print 'sending command:', command
        
#         else:
#             raise Exception('Unknown device: ' + str(dev))

#     # Note: for get_pos and get_vel, conversion from deg to rad occurs inside
#     # udp_feedback_client.py

#     def get_pos(self, dev='IsMore'):
#         if dev == 'ArmAssist':
#             state_names = self.aa_p_state_names
#         elif dev == 'ReHand':
#             state_names = self.rh_p_state_names
#         elif dev == 'IsMore':
#             state_names = self.all_p_state_names
#         else:
#             raise Exception('Unknown device: ' + str(dev))

#         return self.feedback_source.get(n_pts=1, channels=state_names).reshape(-1)

#     def get_vel(self, dev='IsMore'):
#         if dev == 'ArmAssist':
#             state_names = self.aa_v_state_names
#         elif dev == 'ReHand':
#             state_names = self.rh_v_state_names
#         elif dev == 'IsMore':
#             state_names = self.all_v_state_names
#         else:
#             raise Exception('Unknown device: ' + str(dev))

#         return self.feedback_source.get(n_pts=1, channels=state_names).reshape(-1)