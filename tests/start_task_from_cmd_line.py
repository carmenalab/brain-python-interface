from db import dbfunctions
from db.tracker import models

from riglib import experiment

task = models.Task.objects.get(name='test_graphics')
base_class = task.get()
Exp = experiment.make(base_class, feats=[])
#params.trait_norm(Exp.class_traits())
params = {}

seq = models.Sequence.objects.get(id=91)
if issubclass(Exp, experiment.Sequence):
    gen, gp = seq.get()
    sequence = gen(Exp, **gp)
    exp = Exp(sequence, **params)
else:
    exp = Exp(**self.params.params)

exp.start()
