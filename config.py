from ConfigParser import SafeConfigParser
import os

parser = SafeConfigParser()
parser.read(os.path.expandvars('$BMI3D/config'))

recording_system = dict(parser.items('recording_sys'))['make']
