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

task = models.Task.objects.get(name='bmi_control_multi')
base_class = task.get()

from features.generator_features import Autostart
from features.hdf_features import SaveHDF
from features.plexon_features import PlexonBMI, RelayPlexByte

feats = [Autostart, SaveHDF, RelayPlexByte, PlexonBMI]
Exp = experiment.make(base_class, feats=feats)

#params.trait_norm(Exp.class_traits())
params = dict(session_length=10)

seq = models.Sequence.objects.get(id=91)
if issubclass(Exp, experiment.Sequence):
    gen, gp = seq.get()
    sequence = gen(Exp, **gp)
    exp = Exp(sequence, **params)
else:
    raise ValueError('Unknown experiment type')

import MROgraph

colors='edge [color=blue]; node [color=red];'
MROgraph.MROgraph(Exp, filename='%s_mro.png' % task.name, labels=0, caption=True, setup='size="8,6"; ratio=0.7; '+colors)
