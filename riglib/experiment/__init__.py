import os
import time
import random
import threading

import numpy as np

try:
    import traits.api as traits
    import traits.trait_types as trait_types
except ImportError:
    import enthought.traits.api as traits
    import enthought.traits.trait_types as trait_types

import features
import generate

from report import report, print_report
from experiment import Experiment, LogExperiment, Sequence, TrialTypes
from Pygame import Pygame

from tasks import redgreen

featlist = dict(
    autostart=features.Autostart, 
    button=features.Button, 
    button_only=features.ButtonOnly, 
    ignore_correctness=features.IgnoreCorrectness,
    reward_system = features.RewardSystem,
)
genlist = dict(
    endless=generate.endless,
    redgreen_rand=redgreen.randcoords,

    #These are static generators
    trialtypes=generate.sequence,
    redgreen=redgreen.gencoords,
)

typemap = {trait_types.Tuple:tuple}

def make(exp_class, feats=()):
    clslist = tuple(featlist[f] for f in feats if f in featlist)
    clslist = clslist + tuple(f for f in feats if f not in featlist) + (exp_class,)
    return type(exp_class.__name__, clslist, dict())

def consolerun(exp_class, features=(), probs=None, **kwargs):
    Class = make(exp_class, features)
    if probs is None or isinstance(probs, (list, tuple, np.ndarray)):
        gen = generate.endless(Class, probs)
    else:
        gen = probs
    exp = Class(gen, **kwargs)
    exp.start()
    while raw_input().strip() != "q":
        print_report(report(exp))
    exp.end_task()
    print "Waiting to end..."
    exp.join()
    return exp
