
raise ImportError('riglib.loc_config: This module should now be deprecated, replaced by the config.py file!')
import os
hostname = os.popen('hostname').readlines()[0].rstrip()

if hostname == 'arc':
    reward_system_version = 1    
    recording_system = "plexon"
    display_start_pos = "0,0"
elif hostname == 'nucleus':
    # Older Crist reward system (single channel)
    reward_system_version = 0
    display_start_pos = "0,0"
    recording_system = "plexon"
else:
    display_start_pos = "0,0"
    recording_system = "blackrock"
