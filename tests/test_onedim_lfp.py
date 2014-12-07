from db import dbfunctions
from db import json_param
from db.tracker import models

from riglib import experiment
from riglib.stereo_opengl.window import MatplotlibWindow
from features.generator_features import Autostart
from features.hdf_features import SaveHDF

from features.plexon_features import PlexonBMI
from riglib.experiment import generate

from analysis import performance
from tasks.manualcontrolmultitasks import ManualControlMulti
from tasks.bmimultitasks import SimBMIControlMulti
import riglib.bmi.onedim_lfp_decoder as old
from riglib.bmi import clda
from riglib.bmi import train
from analysis import performance



import os
os.environ['DISPLAY'] = ':0'

save = True

task = models.Task.objects.get(name='onedim_lfp')
base_class = task.get()

feats = [SaveHDF, Autostart, PlexonBMI]
Exp = experiment.make(base_class, feats=feats)

#params.trait_norm(Exp.class_traits())
params = dict(session_length=10, plant_visible=True, plant_type='cursor_14x14', 
        rand_start=(0.,0.), max_tries=1)

gen = SimBMIControlMulti.sim_target_seq_generator_multi(8, 1000)
exp = Exp(gen, **params)

kw = dict(control_method='fraction')
n_steps = 10
sf = old.SmoothFilter(n_steps,**kw)
ssm = train.endpt_2D_state_space
units = [[23, 1],[24,1],[25,1]]
decoder = old.One_Dim_LFP_Decoder(exp.neurondata, sf, units, ssm, binlen=0.1, n_subbins=1)

exp.decoder = decoder
ex
self.extractor = self.decoder.extractor_cls(exp.neurondata, **exp.decoder.extractor_kwargs)

exp.init()
exp.run()
