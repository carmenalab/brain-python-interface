import pickle
from riglib.bmi import train
pkl = pickle.load(open('/home/tecnalia/code/bmi3d/tests/ibmi/dec_br.p'))
from ismore.ismore_bmi_lib import StateSpaceArmAssist
import namelist
kin_ext = namelist.kin_extractors[pkl['kin_extractor']]


files = pkl['files']
extractor_cls = pkl['extractor_cls']
extractor_kwargs = pkl['extractor_kwargs']
kin_extractor = kin_ext
ssm = pkl['ssm']
units = pkl['units']
update_rate=0.1
tslice=None
kin_source='task'
pos_key='plant_pos'
vel_key=None

dec = train.train_KFDecoder(pkl['files'], 
					pkl['extractor_cls'], 
					pkl['extractor_kwargs'], 
					kin_ext, 
					StateSpaceArmAssist(),
					pkl['units'],
					update_rate=0.1, 
					tslice=None, 
					kin_source='task', 
					pos_key='plant_pos', 
					vel_key=None)