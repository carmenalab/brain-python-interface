'''Emulates the IsMore ArmAssist application (accepts velocity commands and
sends feedback data over UDP) using the simulated ArmAssist.
'''

import time
import numpy as np
from math import sin, cos
import socket
import select

from riglib.ismore import settings
from utils.constants import *
import pygame

import armassist



# move_automatically = True
move_automatically = False

# keyboard_input_enabled = True
keyboard_input_enabled = False

# how often keyboard input is processed
keyboard_period = .050  # TODO -- decrease this?
ignore_recv_commands = keyboard_input_enabled

control_var = 'pos'  # 'pos' or 'vel'

if keyboard_input_enabled:
    pygame.init()
    screen = pygame.display.set_mode((100,100))


MAX_MSG_LEN = 200  # characters

feedback_freq = 25  # Hz
feedback_period = 1./feedback_freq  # secs


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(settings.ARMASSIST_UDP_SERVER_ADDR)


# create and start ArmAssist object (includes ArmAssist and its PIC)
aa_tstep = 0.005
aa_pic_tstep = 0.01
KP = np.mat([[-10.,   0.,  0.],
             [  0., -20.,  0.],
             [  0.,   0., 20.]])  # P gain matrix
TI = 0.1*np.identity(3)           # I gain matrix

aa = armassist.ArmAssist(aa_tstep, aa_pic_tstep, KP, TI)
aa.daemon = True
aa.start()

starting_pos = settings.starting_pos[['aa_px', 'aa_py', 'aa_ppsi']]
aa._set_wf(np.mat(starting_pos))
print 'setting ArmAssist starting position in sim app'


# uncomment these lines only as a test -- you can run this script by itself
#   and look at the printed feedback packets to verify that that the 
#   appropriate position variables are changing over time based on the
#   velocities set below
# wf_dot_ref = np.mat([0, .2, 0]).T
# aa.update_reference(wf_dot_ref)

t_start = time.time()
t_last_feedback = t_start
t_last_keyboard_iteration = t_start

move_map = {pygame.K_LEFT:  np.array([-1,  0,  0]),
            pygame.K_RIGHT: np.array([ 1,  0,  0]),
            pygame.K_UP:    np.array([ 0,  1,  0]),
            pygame.K_DOWN:  np.array([ 0, -1,  0]),
            pygame.K_o:     np.array([ 0,  0,  1*deg_to_rad]),
            pygame.K_p:     np.array([ 0,  0, -1*deg_to_rad])}

# only start counting after first command is received
n_feedback_packets_sent = 0
received_first_cmd = False

while True:
    if not ignore_recv_commands:
        # check if there is data available to receive before calling recv
        r, _, _ = select.select([sock], [], [], 0)
        if r:  # if the list r is not empty
            command = sock.recv(MAX_MSG_LEN)  # the command string
            print 'received command:', command.rstrip('\r')

            if command == 'ACK\r' or command == 'SetControlMode ArmAssist Global\n':
                pass
            else:
                if not received_first_cmd:
                    received_first_cmd = True
                    t_received_first_cmd = time.time()

                items = command.rstrip('\r').split(' ')
                cmd_id = items[0]
                dev_id = items[1]
                data_fields = items[2:]

                assert dev_id == 'ArmAssist'

                if cmd_id == 'SetSpeed':
                    des_x_vel     = float(data_fields[0]) * mm_to_cm  # convert from mm/s to cm/s
                    des_y_vel     = float(data_fields[1]) * mm_to_cm  # convert from mm/s to cm/s
                    des_z_ang_vel = float(data_fields[2]) * deg_to_rad  # convert from deg/s to rad/s
                    wf_dot_ref = np.mat([des_x_vel, des_y_vel, des_z_ang_vel]).T
                    
                    # expects units of [cm/s, cm/s, rad/s]
                    aa.update_reference(wf_dot_ref)

    
    t = time.time()
    if keyboard_input_enabled and (t - t_last_keyboard_iteration > keyboard_period):
        # move based on keyboard input
        state = aa.get_state()
        if control_var == 'pos':  
            pos = np.array(state['wf']).reshape((3,))

            pressed = pygame.key.get_pressed()
            for m in (move_map[key] for key in move_map if pressed[key]):
                pos += 0.1*m

            pygame.event.pump()

            aa._set_wf(np.mat(pos).reshape(3, 1))
        elif control_var == 'vel':
            vel = np.array(state['wf_dot']).reshape((3,))

            # events = pygame.event.get()
            # for event in events:
            #     if event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_LEFT:
            #             vel[0] -= 0.1
            #         elif event.key == pygame.K_RIGHT:
            #             vel[0] += 0.1
            #         elif event.key == pygame.K_DOWN:
            #             vel[1] -= 0.1
            #         elif event.key == pygame.K_UP:
            #             vel[1] += 0.1
            #         elif event.key == pygame.K_DELETE:
            #             vel[0] = 0.0
            #             vel[1] = 0.0
            #             vel[2] = 0.0
            pressed = pygame.key.get_pressed()
            for m in (move_map[key] for key in move_map if pressed[key]):
                vel += 0.1*m

            # expects units of [cm/s, cm/s, rad/s]
            aa.update_reference(np.mat(vel).reshape(3, 1))

        t_last_keyboard_iteration = t

    t = time.time()
    if t - t_last_feedback > feedback_period:
        state = aa.get_state()
        pos = np.array(state['wf']).reshape((3,))  # units of [cm, cm, rad]
        ts = int(time.time() * 1e6)  # in microseconds

        # convert from cm to mm
        pos[0] *= cm_to_mm
        pos[1] *= cm_to_mm

        # convert from rad to deg
        pos[2] *= rad_to_deg

        freq      = int(1./feedback_period)
        force     = -1
        bar_angle = -1
        ts_aux    = ts

        data_fields = (freq, pos[0], pos[1], pos[2], ts, force, bar_angle, ts_aux)
        feedback = 'Status ArmAssist %d %f %f %f %d %f %f %d\r' % data_fields
        
        # data_fields = (ts, pos[0], pos[1], pos[2], ts_aux, force, bar_angle, -1)
        # feedback = 'Status ArmAssist %d %f %f %f %d %f %f %d\r' % data_fields

        print 'sending feedback:', feedback.rstrip('\r')
        print '\n'

        sock.sendto(feedback, settings.ARMASSIST_UDP_CLIENT_ADDR)

        if received_first_cmd:
            n_feedback_packets_sent += 1
            print '# feedback packets sent:', n_feedback_packets_sent
            print 'packets/sec:', n_feedback_packets_sent / (time.time() - t_received_first_cmd)

        t_last_feedback = t

        if move_automatically and received_first_cmd:
            t = time.time() - t_start
            wf_dot_ref = np.mat([1 * sin(0.4 * t), 
                                 1 * cos(0.4 * t), 
                                 0]).T #.2 * sin(0.1 * t) * cos(0.1 * t)]).T
            aa.update_reference(wf_dot_ref)


# stop ArmAssist process
aa.stop()
