import os
hostname = os.popen('hostname').readlines()[0].rstrip()

if hostname == 'arc':
    reward_system_version = 1
elif hostname == 'nucleus':
    # Older Crist reward system (single channel)
    reward_system_version = 0
else:
    pass
