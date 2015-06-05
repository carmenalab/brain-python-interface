'''Emulates the IsMore ReHand application (accepts velocity commands and
sends feedback data over UDP) using the simulated ReHand.
'''

import time
import numpy as np
from math import sin, cos
import rehand
import socket
import select

from riglib.ismore import settings
from utils.constants import *


MAX_MSG_LEN = 200  # characters

feedback_freq = 200  # Hz
feedback_period = 1./feedback_freq  # secs


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(settings.REHAND_UDP_SERVER_ADDR)


# create and start ReHand object
rh = rehand.ReHand(tstep=0.005)
rh.daemon = True
rh.start()

starting_pos = settings.starting_pos[['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']]
rh._set_pos(np.mat(starting_pos))
print 'setting ReHand starting position in sim app'

t_last_feedback = time.time()

# only start counting after first command is received
n_feedback_packets_sent = 0
received_first_cmd = False

while True:
    # check if there is data available to receive before calling recv
    r, _, _ = select.select([sock], [], [], 0)
    if r:  # if the list r is not empty
        command = sock.recv(MAX_MSG_LEN) # the command string
        print 'received command:', command.rstrip('\r')
        if not received_first_cmd:
            received_first_cmd = True
            t_received_first_cmd = time.time()
            
        items = command.rstrip('\r').split(' ')
        cmd_id = items[0]
        dev_id = items[1]
        data_fields = items[2:]

        print 'cmd_id', cmd_id
        print 'dev_id', dev_id
        print 'equal to ReHand?', dev_id == 'ReHand'
        print 'len(dev_id)', len(dev_id)
        print command
        print items

        assert dev_id == 'ReHand'

        if cmd_id == 'SetSpeed':
            des_thumb_vel = float(data_fields[0]) * deg_to_rad  # convert from deg/s to rad/s
            des_index_vel = float(data_fields[1]) * deg_to_rad  # convert from deg/s to rad/s
            des_fing3_vel = float(data_fields[2]) * deg_to_rad  # convert from deg/s to rad/s
            des_prono_vel = float(data_fields[3]) * deg_to_rad  # convert from deg/s to rad/s
            des_vel = np.mat([des_thumb_vel, des_index_vel, des_fing3_vel, des_prono_vel]).T

            rh.set_vel(des_vel)

    t = time.time()
    if t - t_last_feedback > feedback_period:
        # print 'sending feedback'
        state = rh.get_state()
        pos = np.array(state['pos']).reshape((4,))
        vel = np.array(state['vel']).reshape((4,))
        ts = int(time.time() * 1e6)  # in microseconds

        # convert from rad to deg (and rad/s to deg/s)
        pos *= rad_to_deg
        vel *= rad_to_deg

        freq = -1

        data_fields = (freq, 
                       vel[0], pos[0], 0, 
                       vel[1], pos[1], 0, 
                       vel[2], pos[2], 0, 
                       vel[3], pos[3], 0,
                       ts)

        feedback = 'ReHand Status %f %f %f %f %f %f %f %f %f %f %f %f %f %d\r' % data_fields 
        print 'sending feedback:', feedback.rstrip('\r')
        print '\n'

        sock.sendto(feedback, settings.REHAND_UDP_CLIENT_ADDR)

        if received_first_cmd:
            n_feedback_packets_sent += 1
            print '# feedback packets sent:', n_feedback_packets_sent
            print 'packets/sec:', n_feedback_packets_sent / (time.time() - t_received_first_cmd)

        t_last_feedback = t


# stop ReHand process
rh.stop()
