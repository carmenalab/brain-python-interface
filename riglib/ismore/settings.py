import pandas as pd
from utils.constants import *

# send SetSpeed commands to these addresses
armassist_udp_server = ('127.0.0.1', 5001)
# rehand_udp_server    = ('127.0.0.1', 5000)
rehand_udp_server    = ('192.168.137.6', 5000)

# receive feedback data on these addresses
armassist_udp_client = ('127.0.0.1', 5002)
# rehand_udp_client    = ('127.0.0.1', 5003)
rehand_udp_client    = ('192.168.137.2', 5003)

starting_pos = pd.Series({
    'aa_px':     43,                # cm
    'aa_py':     18.,               # cm
    'aa_ppsi':    0.,               # rad
    'rh_pthumb': 30. * deg_to_rad,  # rad
    'rh_pindex': 30. * deg_to_rad,  # rad
    'rh_pfing3': 30. * deg_to_rad,  # rad
    'rh_pprono':  2. * deg_to_rad,  # rad
})
