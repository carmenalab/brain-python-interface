'''
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

class DataFile(traits.Instance):
    def __init__(self, *args, **kwargs):
        if 'bmi3d_query_kwargs' in kwargs:
            self.bmi3d_query_kwargs = kwargs['bmi3d_query_kwargs']

        super(DataFile, self).__init__(*args, **kwargs)

class OptionsList(traits.Enum):
    def __init__(self, *args, **kwargs):
        if 'bmi3d_input_options' in kwargs:
            self.bmi3d_input_options = kwargs['bmi3d_input_options']

        super(OptionsList, self).__init__(*args, **kwargs)

traits.DataFile = DataFile
traits.OptionsList = OptionsList



import experiment
import generate
import report
from experiment import Experiment, LogExperiment, Sequence, TrialTypes, FSMTable, StateTransitions

try:
    from Pygame import Pygame
except:
    import warnings
    warnings.warn('riglib/experiment/__init__.py: could not import Pygame (note capital P)')
    Pygame = object

def make(exp_class, feats=()):
    '''
    Creates a class which inherits from a base experiment class as well as a set of optional features.
    This function is a *metafunction* as it returns a custom class construction.

    Parameters
    ----------
    exp_class : class
        Base class containing the finite state machine of the task
    feats : iterable of classes
        Additional classes from which to also inherit

    Returns
    -------
    class
        New class which inherits from the base 'exp_class' and the selected 'feats'
    '''
    # construct the class list to define inheritance order for the custom task
    # inherit from the features first, then the base class
    clslist = tuple(feats) + (exp_class,)

    # return custom class
    return type(exp_class.__name__, clslist, dict())

def make_and_inst(exp_class, features=(), probs=None, **kwargs):
    """
    Instantiate a task from the shell
    Docstring

    Parameters
    ----------

    Returns
    -------
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
    Docstring

    Parameters
    ----------

    Returns
    -------
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
