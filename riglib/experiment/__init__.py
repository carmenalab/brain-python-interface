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
import generate

import report
from experiment import Experiment, LogExperiment, Sequence, TrialTypes
from Pygame import Pygame

def make(exp_class, feats=()):
    clslist = tuple(feats) + (exp_class,)
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
