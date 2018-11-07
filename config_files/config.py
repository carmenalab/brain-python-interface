'''
Module to read the 'config' text file for rig-specific configuration
'''
from ConfigParser import SafeConfigParser
import os
import sys

## Hack getattr: http://stackoverflow.com/questions/2447353/getattr-on-a-module
class Config(object):
    def __init__(self):
        parser = SafeConfigParser()
        self.parser = parser
        if not os.path.exists(os.path.expandvars('$BMI3D/config_files/config')):
            raise ValueError("config.py cannot find 'config' file. Check that it exists and that the environment variable $BMI3D is defined!")

        self.parser.read(os.path.expandvars('$BMI3D/config_files/config'))

        self.recording_system = dict(parser.items('recording_sys'))['make']
        self.data_path = dict(parser.items('db_config_default'))['data_path']

        self.window_start_x = dict(parser.items('graphics'))['window_start_x']
        self.window_start_y = dict(parser.items('graphics'))['window_start_y']
        self.display_start_pos = '%s,%s' % (self.window_start_x, self.window_start_y)

        self.reward_system_version = int(dict(parser.items('reward_sys'))['version'])        
        self.log_dir = '/home/lab/code/bmi3d/log'
        self.plexon_ip = dict(parser.items('plexon IP address'))['addr']
        self.plexon_port = dict(parser.items('plexon IP address'))['port']

        try:
            self.hdf_update_rate_hz = int(dict(parser.items('update_rates'))['hdf_hz'])
        except:
            self.hdf_update_rate_hz = 60.

    def __getattr__(self, attr):
        try:
            return dict(self.parser.items(attr))
        except:
            raise AttributeError("config.py: Attribute '%s' not found in config file!" % attr)

config = Config()
