import os
import time
import random
import threading
import numpy as np

try:
    import traits.api as traits
except ImportError:
    import enthought.traits.api as traits

import features
from report import report, print_report
from experiment import Experiment, LogExperiment, Sequence, TrialTypes
from Pygame import Pygame

def make_experiment(exp_class, feats=()):
    allfeats = dict(
        button=features.Button,
        button_only=features.ButtonOnly,
        autostart=features.Autostart,
        ignore_correctness=features.IgnoreCorrectness
    )
    clslist = tuple(allfeats[f] for f in feats if f in allfeats)
    clslist = clslist + tuple(f for f in feats if f not in allfeats) + (exp_class,)
    return type(exp_class.__name__, clslist, dict())

def consolerun(exp_class, features=(), **kwargs):
    Class = make_experiment(exp_class, features)
    exp = Class(**kwargs)
    exp.start()
    while raw_input().strip() != "q":
        print_report(report(exp))
    exp.end_task()
    print "Waiting to end..."
    exp.join()
    return exp