'''Needs docs
Experiment constructors. 'experiment' instances are the combination of 
a task and a list of features.  Rather than have a separate class for
all the possible combinations of tasks and features, a custom class for
the experiment is created programmatically using 'type'. The created class 
has methods of the base task as well as all the selected features. 
'''
import numpy as np

try:
    import traits.api as traits
except ImportError:
    import enthought.traits.api as traits

import generate

import report
import features
from experiment import Experiment, LogExperiment, Sequence, TrialTypes

try:
    from Pygame import Pygame
except:
    import warnings
    warnings.warn('riglib/experiment/__init__.py: could not import Pygame (note capital P)')
    Pygame = object

def make(exp_class, feats=()):
    f = open('/home/helene/code/bmi3d/log/ajax_task_startup', 'a')
    f.write('experiment.make: %s, %s\n' % (exp_class, feats))
    f.close()
    # construct the class list to define inheritance order for the custom task
    # inherit from the features first, then the base class
    clslist = tuple(feats) + (exp_class,) 

    # return custom class
    return type(exp_class.__name__, clslist, dict())

def make_and_inst(exp_class, features=(), probs=None, **kwargs):
    """
    Instantiate a task from the shell
    """
    # Construct the custom task instance as a 
    # as a combination of the base task and the selected features
    Class = make(exp_class, features)

    # instantiate generator for task trials
    if probs is None or isinstance(probs, (list, tuple, np.ndarray)):
        gen = generate.endless(Class, probs)
    else:
        gen = probs

    # instantiate task
    exp = Class(gen, **kwargs)
    return exp

def consolerun(exp_class, features=(), probs=None, **kwargs):
    """
    Instantiate a task from the shell
    """
    # Construct the custom task instance as a 
    # as a combination of the base task and the selected features
    Class = make(exp_class, features)

    # instantiate generator for task trials
    if probs is None or isinstance(probs, (list, tuple, np.ndarray)):
        gen = generate.endless(Class, probs)
    else:
        gen = probs

    # instantiate task
    exp = Class(gen, **kwargs)
    exp.start()

    # run until 'q' is pressed on the keyboard
    while raw_input().strip() != "q":
        print_report(report(exp))

    exp.end_task()
    print "Waiting to end..."
    exp.join()
    return exp
