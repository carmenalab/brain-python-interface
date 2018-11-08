import socket
import select
import time
import numpy as np

from ismore import udp_feedback_client, settings
from utils.constants import *

MAX_MSG_LEN = 200
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# input address of this application (address on which to receive data)
sock.bind(settings.ARMASSIST_UDP_CLIENT_ADDR)

def recv_feedback():
    rlist, _, _ = select.select([sock], [], [], 1)
    if rlist:  # if rlist is not empty
        feedback = sock.recv(MAX_MSG_LEN)
        return feedback
    else:
        return None

px_vec   = []
py_vec   = []
ppsi_vec = []

# print out received feedback and estimated feedback rate in a loop
t_start = time.time()
recv_packet_count = 0
while True:
    feedback = recv_feedback()
    if feedback is not None:
        print(feedback)

        # TODO -- replace code below with a call to 
        # udp_feedback_client.ArmAssistData.process_received_feedback()

        items = feedback.rstrip('\r').split(' ')
        
        cmd_id      = items[0]
        dev_id      = items[1]
        data_fields = items[2:]
        
        assert cmd_id == 'Status'
        assert dev_id == 'ArmAssist'
        assert len(data_fields) == 8

        freq = float(data_fields[0])                    # Hz

        # position data
        px   = float(data_fields[1]) * mm_to_cm         # cm
        py   = float(data_fields[2]) * mm_to_cm         # cm
        ppsi = float(data_fields[3]) * deg_to_rad       # rad
        ts   = int(data_fields[4])   * us_to_s          # sec

        px_vec.append(px)
        py_vec.append(py)
        ppsi_vec.append(ppsi)

        print('min:', np.min(np.array(px_vec)))
        print('max:', np.max(np.array(px_vec)))
        print('std:', np.std(np.array(px_vec)))

        print('min:', np.min(np.array(py_vec)))
        print('max:', np.max(np.array(py_vec)))
        print('std:', np.std(np.array(py_vec)))

        print('min:', np.min(np.array(ppsi_vec)))
        print('max:', np.max(np.array(ppsi_vec)))
        print('std:', np.std(np.array(ppsi_vec)))

        recv_packet_count += 1
        t_elapsed = time.time() - t_start
        print(recv_packet_count / t_elapsed)
