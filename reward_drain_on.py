#!/usr/bin/python

import os
#export BMI3D=/home/helene/code/bmi3d
os.environ['BMI3D'] = os.path.join(os.path.expandvars('$HOME/code/bmi3d'))

from riglib import reward
import time


r = reward.Basic()
time.sleep(.5)
r.drain(600)
