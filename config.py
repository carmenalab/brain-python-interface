from ConfigParser import SafeConfigParser
import os
import sys

## Hack getattr: http://stackoverflow.com/questions/2447353/getattr-on-a-module
class Config:
	def __init__(self):
		parser = SafeConfigParser()
		self.parser = parser
		self.parser.read(os.path.expandvars('$BMI3D/config'))

		self.recording_system = dict(parser.items('recording_sys'))['make']
		self.data_path = dict(parser.items('db_config_default'))['data_path']

		self.window_start_x = dict(parser.items('graphics'))['window_start_x']
		self.window_start_y = dict(parser.items('graphics'))['window_start_y']
		self.display_start_pos = '%s,%s' % (self.window_start_x, self.window_start_y)

		self.reward_system_version = int(dict(parser.items('reward_sys'))['version'])		
		self.log_dir = os.path.expandvars('$BMI3D/log')

	def __getattr__(self, attr):
		if attr in self.parser.sections():
			return dict(self.parser.items(attr))
		else:
			return super(Config, self).__getattr__(attr)

config = Config()
