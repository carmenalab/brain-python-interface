from riglib import bmi
from riglib.bmi import extractor
import numpy as np

kinarm_bands = []
for i in np.arange(0,100,10):
	kinarm_bands.extend([[i, i+10]])
kinarm_bands.extend([[25, 40],[40, 55], [65, 90], [2, 100]])

class SmoothFilter(object):
	'''Moving Avergae Filter used in 1D LFP control:
	x_{t} = a0*x_{t} + a1*x_{t-1} + a2*x_{t-2} + ...

	Parameters
    ----------
    A: np.array of shape (N, )
        Weights for previous states
    X: np. array of previous states (N, )
	'''
	model_attrs = []

	def __init__(self, n_steps, **kwargs):
		self.n_steps = n_steps
		self.A = np.ones(( n_steps, ))/float(n_steps)
		
	def get_mean(self):
		return np.array(self.state.mean).ravel()

	def _init_state(self, init_state=None,**kwargs):
		if init_state is None:
			self.X = np.zeros(( self.n_steps, ))
		
		elif init_state is 'average':
			if control_method == 'fraction':
				mn = np.mean(np.array(kwargs['frac_lim']))
			elif control_method == 'total_power':
				mn = np.mean(np.array(kwargs['pwr_mean']))
			self.X = np.zeros(( self.n_steps )) + mn


	def __call__(self, obs, **kwargs):
		self.state = self._mov_avg(self.state, obs, **kwargs)

	def _mov_avg(self, obs):
		self.X = np.hstack(( self.X[1:], obs ))
		return np.dot(self.X, self.A)

class One_Dim_LFP_Decoder(bmi.Decoder):

	def __init__(self, *args, **kwargs):
		
		bands = kinarm_bands
		control_method='fraction'
		no_log=True
		
		super(One_Dim_LFP_Decoder, self).__init__(*args, **kwargs)
		
		if no_log:
			kw = dict(no_log=no_log)

		#For now: 
		source = None
		self.extractor_cls = extractor.LFPMTMPowerExtractor(source,self.units,bands=bands,**kw)
		self.extractor_kwargs = self.extractor_cls.extractor_kwargs
		
def train_1d_LFP_dec(extractor_cls, extractor_kwargs, units,control_method='fraction'):
	kw = dict(control_method='fraction')
	sf = SmoothFilter(n_steps,**kw)
	decoder = One_Dim_LFP_Decoder(sf, units, ssm, binlen=0.1, n_subbins=1)
	decoder.extractor_cls = extractor_cls
	decoder.extractor_kwargs = extractor_kwargs
	return decoder






