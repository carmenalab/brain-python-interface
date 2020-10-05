# '''
# Module to read the 'config' text file for rig-specific configuration
# '''
# from configparser import SafeConfigParser
# import os
# import sys

# ## Hack getattr: http://stackoverflow.com/questions/2447353/getattr-on-a-module
# class Config(object):
#     config_fname = os.path.join(os.path.dirname(__file__), "config")
#     def __init__(self):
#         self.log_path = os.path.join(os.path.dirname(__file__), "../log")
#         self.log_dir = self.log_path

#         if not os.path.exists(self.log_path):
#             os.mkdir(self.log_path)

#         self.window_start_x = 0
#         self.window_start_y = 0
#         self.display_start_pos = '%s,%s' % (self.window_start_x, self.window_start_y)
#         try:
#             parser = SafeConfigParser()
#             self.parser = parser

#             if not os.path.exists(self.config_fname):
#                 raise ValueError("config.py cannot find 'config' file at expected location %s" % self.config_fname)

#             self.parser.read(self.config_fname)

#             self.recording_system = dict(parser.items('recording_sys'))['make']
#             self.data_path = dict(parser.items('db_config_default'))['data_path']

#             self.window_start_x = dict(parser.items('graphics'))['window_start_x']
#             self.window_start_y = dict(parser.items('graphics'))['window_start_y']
#             self.display_start_pos = '%s,%s' % (self.window_start_x, self.window_start_y)

#             self.reward_system_version = int(dict(parser.items('reward_sys'))['version'])
#             self.plexon_ip = dict(parser.items('plexon IP address'))['addr']
#             self.plexon_port = dict(parser.items('plexon IP address'))['port']

#             try:
#                 self.hdf_update_rate_hz = int(dict(parser.items('update_rates'))['hdf_hz'])
#             except:
#                 self.hdf_update_rate_hz = 60.
#         except:
#             pass

#     def __getattr__(self, attr):
#         try:
#             return dict(self.parser.items(attr))
#         except:
#             raise AttributeError("config.py: Attribute '%s' not found in config file!" % attr)

#     def write_to_config_file(self, config_data):
#         config_fh = open(os.path.expandvars(self.config_fname), 'w')

#         for system_name, system_opts in list(config_data.items()):
#             config_fh.write('[%s]\n' % system_name)
#             for option, opt_val in list(system_opts.items()):
#                 config_fh.write('%s = %s\n' % (option, opt_val))
#             config_fh.write('\n')

#         config_fh.close()
#         return self.config_fname


# config = Config()
