'''
Features for interacting with neuropixels
'''
import time
import os
import numpy as np
from open_ephys.control import OpenEphysHTTPServer

IP_neuropixel = '10.155.205.108'
gui = OpenEphysHTTPServer(IP_neuropixel)

gui.acquire(10)
