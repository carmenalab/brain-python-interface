import pandas as pd

from ismore import brainamp_channel_lists
from ismore.common_state_lists import *
from utils.constants import *


#PLANT_TYPE = 'ArmAssist'
#PLANT_TYPE = 'ReHand'
PLANT_TYPE = 'IsMore'

USE_NONUDP_PLANTS = False
USE_LFC_ASSISTER  = False
USE_OFC_LEARNER   = False
USE_RML           = True


#### BrainAmp-related settings ####

VERIFY_BRAINAMP_DATA_ARRIVAL = True
# print warning if EMG data doesn't arrive or stops arriving for this long
VERIFY_BRAINAMP_DATA_ARRIVAL_TIME = 1  # secs

# the channels that the Python code will receive, make available to the task,
#   and save into the HDF file
# (received data on other BrainAmp channels will simply be discarded)
BRAINAMP_CHANNELS = brainamp_channel_lists.emg

###################################


# send SetSpeed commands to server addresses
# receive feedback data on client addresses
ARMASSIST_UDP_SERVER_ADDR = ('127.0.0.1', 5001)
ARMASSIST_UDP_CLIENT_ADDR = ('127.0.0.1', 5002)
REHAND_UDP_SERVER_ADDR    = ('127.0.0.1', 5000)
REHAND_UDP_CLIENT_ADDR    = ('127.0.0.1', 5003)

VERIFY_PLANT_DATA_ARRIVAL = True
# print warning if plant data doesn't arrive or stops arriving for this long
VERIFY_PLANT_DATA_ARRIVAL_TIME = 1  # secs

DONNING_SAME_AS_STARTING_POSITION = True

WATCHDOG_ENABLED = False
WATCHDOG_TIMEOUT = 1000  # ms


MAT_SIZE = [85, 95]  # cm


starting_pos = pd.Series(0.0, ismore_pos_states)

if MAT_SIZE == [42, 30]:           # smallest mat
    starting_pos['aa_px']   = 21.  # cm
    starting_pos['aa_py']   = 15.  # cm
    starting_pos['aa_ppsi'] =  0.  # rad      
elif MAT_SIZE == [71, 51]:         # small mat
    starting_pos['aa_px']   = 37.  # cm
    starting_pos['aa_py']   =  4.  # cm
    starting_pos['aa_ppsi'] =  0.  # rad    
elif MAT_SIZE == [85, 95]:         # large mat
    starting_pos['aa_px']   = 40.  # cm
    starting_pos['aa_py']   = 18.  # cm
    starting_pos['aa_ppsi'] =  0.  # rad
else:
    raise Exception('Unknown MAT_SIZE in riglib/ismore/settings.py!')

starting_pos['rh_pthumb'] = 30. * deg_to_rad  # rad
starting_pos['rh_pindex'] = 30. * deg_to_rad  # rad
starting_pos['rh_pfing3'] = 30. * deg_to_rad  # rad
starting_pos['rh_pprono'] = 30. * deg_to_rad  # rad

if DONNING_SAME_AS_STARTING_POSITION:
    donning_position = starting_pos[rh_pos_states]
else:
    donning_position = pd.Series(0.0, rh_pos_states)
    donning_pos['rh_pthumb'] = 30. * deg_to_rad  # rad
    donning_pos['rh_pindex'] = 30. * deg_to_rad  # rad
    donning_pos['rh_pfing3'] = 30. * deg_to_rad  # rad
    donning_pos['rh_pprono'] = 40. * deg_to_rad  # rad
