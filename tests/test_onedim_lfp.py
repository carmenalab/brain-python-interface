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

task = models.Task.objects.get(name='lfp_mod')
#task = models.Task.objects.get(name='manual_control_multi')

base_class = task.get()

feats = [SaveHDF, Autostart, PlexonBMI]
Exp = experiment.make(base_class, feats=feats)

#params.trait_norm(Exp.class_traits())
params = dict(session_length=10, plant_visible=True, plant_type='cursor_14x14', 
        rand_start=(0.,0.), max_tries=1)

gen = SimBMIControlMulti.sim_target_seq_generator_multi(8, 1000)
exp = Exp(gen, **params)

import pickle
decoder = pickle.load(open('/storage/decoders/cart20141206_06_test_lfp1d2.pkl'))
exp.decoder = decoder

exp.init()

exp.run()
