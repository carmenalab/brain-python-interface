#!/usr/bin/python
'''
Executable script to create the configuration file for the BMI3D code, a text file called '$BMI3D/config/config'
'''
import os
import sys
from collections import OrderedDict

stuff = OrderedDict()

stuff['reward_sys'] = dict(version=0)
stuff['recording_sys'] = dict(make='plexon', mount_point='/storage/plexon')
stuff['graphics'] = dict(window_start_x=0, window_start_y=0)
stuff['backup_root'] = dict(root='/backup')
stuff['plexon IP address'] = dict(addr='10.0.0.13', port=6000)
stuff['update_rates'] = dict(hdf_hz=60)

# Add an optional commandline flag to make setup non-interactive
use_defaults = '-y' in sys.argv or '--use-defaults' in sys.argv

from db import settings
databases = list(settings.DATABASES.keys())

for db_name in databases:
    stuff[f'db_config_{db_name}'] = dict(data_path='/storage')

config_filename = '$BMI3D/config/config'
config_fh = open(os.path.expandvars(config_filename), 'w')

for system_name, system_opts in list(stuff.items()):
    config_fh.write(f'[{system_name}]\n')
    print(system_name)
    for option, default in list(system_opts.items()):

        if use_defaults:
            print(f"  Using default ({default}) for {option}")
            opt_val = default
        else:
            opt_val = input(f"  Enter value for '{option}' (default={default}): ")
            if opt_val == '':
                opt_val = default

        config_fh.write(f'{option} = {opt_val}\n')
    config_fh.write('\n')
    print() 

config_fh.close()
