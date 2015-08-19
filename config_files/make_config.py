#!/usr/bin/python
'''
Create the configuration file for the BMI3D code
'''
import os
from collections import OrderedDict

stuff = OrderedDict()

stuff['reward_sys'] = dict(version=0)
stuff['recording_sys'] = dict(make='plexon', mount_point='/storage/plexon')
stuff['graphics'] = dict(window_start_x=0, window_start_y=0)
stuff['backup_root'] = dict(root='/backup')

from db import settings
databases = settings.DATABASES.keys()

for dbname in databases:
    stuff['db_config_%s' % dbname] = dict(data_path='/storage')

config_filename = '$BMI3D/config_files/config'
config_fh = open(os.path.expandvars(config_filename), 'w')

for system_name, system_opts in stuff.items():
    config_fh.write('[%s]\n' % system_name)
    print system_name
    for option, default in system_opts.items():
        print option, default
        opt_val = raw_input("Enter value for '%s' (default=%s): " % (option, str(default)))
        if opt_val == '':
            opt_val = default
        config_fh.write('%s = %s\n' % (option, opt_val))
    config_fh.write('\n')
    print 

config_fh.close()
