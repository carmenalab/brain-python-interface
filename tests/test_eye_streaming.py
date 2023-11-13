import socket
import re
import numpy as np
from riglib.oculomatic import System
import time

UDP_IP = ""
UDP_PORT = 11999

# Format is "frame_count&le_x&le_y&le_diam&re_x&re_y&re_diam"

def test_streaming_direct():
    sock = socket.socket(socket.AF_INET, # Internet
                        socket.SOCK_DGRAM) # UDP
    sock.bind((UDP_IP, UDP_PORT))

    while True:
        data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        items = data.decode('utf-8').split('&')
        frame_count = int(items[0])
        try:
            le_x = float(items[1])
        except:
            le_x = np.nan
        try:
            le_y = float(items[2])
        except:
            le_y = np.nan
        le_diam = int(items[3])
        try:
            re_x = float(items[4])
        except:
            re_x = np.nan
        try:
            re_y = float(items[5])
        except:
            re_y = np.nan
        re_diam = int(items[6])

        print(f"frame {frame_count:0>8d}: left eye ({le_x:+06.2f}, {le_y:+06.2f}) {le_diam:0>5d}, right eye ({re_x:+06.2f}, {re_y:+06.2f}) {re_diam:0>5d}")

        
def test_streaming_class():
    from riglib import source
    eyedata = source.DataSource(System)
    eyedata.start()
    time.sleep(0.5)

    data = eyedata.get()
    eyedata.stop()

    print(data)

test_streaming_class()

