from ConfigParser import SafeConfigParser
import os

parser = SafeConfigParser()
parser.read(os.path.expandvars('$BMI3D/config'))

recording_system = dict(parser.items('recording_sys'))['make']
data_path = dict(parser.items('db_config_default'))['data_path']

db_config_default = dict(parser.items('db_config_default'))
db_config_bmi3d = dict(parser.items('db_config_bmi3d'))
db_config_exorig = dict(parser.items('db_config_exorig'))
