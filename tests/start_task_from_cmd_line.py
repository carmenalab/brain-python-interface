'''
Test script to run the visual feedback task from the command line
'''
from db import dbfunctions
from db.tracker import models

from riglib import experiment
from riglib.experiment import features

from tasks import generatorfunctions as genfns

# Tell linux to use Display 0 (the monitor physically attached to the 
# machine. Otherwise, if you are connected remotely, it will try to run 
# the graphics through SSH, which doesn't work for some reason.
import os
os.environ['DISPLAY'] = ':0'

task = models.Task.objects.get(name='visual_feedback_multi')
base_class = task.get()

feats = [features.SaveHDF]
Exp = experiment.make(base_class, feats=feats)

#params.trait_norm(Exp.class_traits())
params = dict(session_length=10, arm_visible=True, arm_class='RobotArmGen2D')

if issubclass(Exp, experiment.Sequence):
    gen = genfns.sim_target_seq_generator_multi(8, 1000)
    exp = Exp(gen, **params)
else:
    exp = Exp(**params)

exp.start()
