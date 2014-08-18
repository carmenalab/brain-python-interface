#!/usr/bin/python
'''
Create the configuration file for the BMI3D code
'''

stuff = dict(reward_sys=dict(version=0),
             recording_sys=dict(make='plexon', mount_point='/storage/plexon'),
             graphics=dict(window_start_x=0, window_start_y=0),
             database=dict(db='/home/helene/code/bmi3d/db/db.sql'))

# TODO get this from command line?
config_filename = 'config'
config_fh = open(config_filename, 'w')

for system_name, system_opts in stuff.items():
    config_fh.write('[%s]\n' % system_name)
    print system_name
    for option, default in system_opts.items():
        print option, default
        opt_val = raw_input("Enter value for '%s' (default=%s)" % (option, str(default)))
        if opt_val == '':
            opt_val = default
        config_fh.write('%s = %s\n' % (option, opt_val))
    config_fh.write('\n')
    print 

config_fh.close()
