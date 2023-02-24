'''
Features for interacting with neuropixels
'''
import time
import os
import numpy as np
from open_ephys.control import OpenEphysHTTPServer

gui = OpenEphysHTTPServer('10.155.205.108')

gui.idle()
print(12)