'''
Test script to run the visual feedback task from the command line
'''
from db import dbfunctions
from db.tracker import models

from riglib import experiment
from riglib.experiment import features

# Tell linux to use Display 0 (the monitor physically attached to the 
# machine. Otherwise, if you are connected remotely, it will try to run 
# the graphics through SSH, which doesn't work for some reason.
import os
os.environ['DISPLAY'] = ':0'

task = models.Task.objects.get(name='visual_feedback_multi')
base_class = task.get()

feats = [features.Autostart, features.SaveHDF, features.RewardSystem]
Exp = experiment.make(base_class, feats=feats)

#params.trait_norm(Exp.class_traits())
params = dict(session_length=10)

seq = models.Sequence.objects.get(id=2)
if issubclass(Exp, experiment.Sequence):
    gen, gp = seq.get()
    sequence = gen(Exp, **gp)
    exp = Exp(sequence, **params)
else:
    raise ValueError('Unknown experiment type')

exp.start()
