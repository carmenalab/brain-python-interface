'''
Module to read the 'config' text file for rig-specific configuration
'''
from configparser import SafeConfigParser
import os
import sys

## Hack getattr: http://stackoverflow.com/questions/2447353/getattr-on-a-module
class Config(object):
    def __init__(self):
        try:
            parser = SafeConfigParser()
            self.parser = parser
            config_fname = os.path.join(os.path.dirname(__file__), "config")
            if not os.path.exists(config_fname):
                raise ValueError("config.py cannot find 'config' file at expected location %s" % config_fname)

            self.parser.read(config_fname)

            self.recording_system = dict(parser.items('recording_sys'))['make']
            self.data_path = dict(parser.items('db_config_default'))['data_path']

            self.window_start_x = dict(parser.items('graphics'))['window_start_x']
            self.window_start_y = dict(parser.items('graphics'))['window_start_y']
            self.display_start_pos = '%s,%s' % (self.window_start_x, self.window_start_y)

            self.reward_system_version = int(dict(parser.items('reward_sys'))['version'])        
            self.log_dir = '/home/lab/code/bmi3d/log'
            self.plexon_ip = dict(parser.items('plexon IP address'))['addr']
            self.plexon_port = dict(parser.items('plexon IP address'))['port']

            self.log_path = "/home/suraj/code/bmi3d/log"

            try:
                self.hdf_update_rate_hz = int(dict(parser.items('update_rates'))['hdf_hz'])
            except:
                self.hdf_update_rate_hz = 60.
        except:
            self.log_path = ""

    def __getattr__(self, attr):
        try:
            return dict(self.parser.items(attr))
        except:
            raise AttributeError("config.py: Attribute '%s' not found in config file!" % attr)

config = Config()
