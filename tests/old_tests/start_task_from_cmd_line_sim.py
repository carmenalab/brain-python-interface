'''
Test script to run the visual feedback task from the command line
'''
from ..db import dbfunctions
from ..db.tracker import models

from riglib import experiment
from ..features.generator_features import Autostart
from ..features.hdf_features import SaveHDF
from ..features.plexon_features import PlexonBMI

from tasks import generatorfunctions as genfns
from analysis import performance

# Tell linux to use Display 0 (the monitor physically attached to the 
# machine. Otherwise, if you are connected remotely, it will try to run 
# the graphics through SSH, which doesn't work for some reason.
import os
os.environ['DISPLAY'] = ':0'

task = models.Task.objects.get(name='clda_kf_ofc_tentacle_rml_trial')
base_class = task.get()

feats = [SaveHDF, PlexonBMI, Autostart]
Exp = experiment.make(base_class, feats=feats)

#params.trait_norm(Exp.class_traits())
params = dict(session_length=30, arm_visible=True, arm_class='RobotArmGen2D', 
        assist_level=(2., 2.), assist_time=60., rand_start=(0.,0.), max_tries=1)

gen = genfns.sim_target_seq_generator_multi(8, 1000)
exp = Exp(gen, **params)

exp.decoder = performance._get_te(3979).decoder

exp.start()
