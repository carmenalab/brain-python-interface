import os
hostname = os.popen('hostname').readlines()[0].rstrip()

if hostname == 'arc':
    reward_system_version = 1
    display_start_pos = "0,0"
elif hostname == 'nucleus':
    # Older Crist reward system (single channel)
    reward_system_version = 0
    display_start_pos = "0,0"
else:
    pass
