import socket
import select
import time

from ismore import settings


MAX_MSG_LEN = 200
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# input address of this application (address to receive from)
sock.bind(settings.REHAND_UDP_CLIENT_ADDR)

def recv_feedback():
    rlist, _, _ = select.select([sock], [], [], 1)
    if rlist:  # if rlist is not empty
        feedback = sock.recv(MAX_MSG_LEN)
        return feedback
    else:
        return None

# print out received feedback and estimated feedback rate in a loop
t_start = time.time()
recv_packet_count = 0
while True:
    feedback = recv_feedback()
    if feedback is not None:
        print(feedback)

        recv_packet_count += 1
        t_elapsed = time.time() - t_start
        print(recv_packet_count / t_elapsed)
