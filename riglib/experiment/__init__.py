'''
Experiment constructors. 'Experiment' instances are the combination of
a task and a list of features.  Rather than have a separate class for
all the possible combinations of tasks and features, a custom class for
the experiment is created programmatically using 'type'. The created class
has methods of the base task as well as all the selected features.
'''
import numpy as np

##################
##### Traits #####
##################
try:
    import traits.api as traits
except ImportError:
    import enthought.traits.api as traits


class InstanceFromDB(traits.Instance):
    def __init__(self, *args, **kwargs):
        if 'bmi3d_db_model' in kwargs:
            self.bmi3d_db_model = kwargs['bmi3d_db_model']
        else:
            raise ValueError("If using trait 'InstanceFromDB', must specify bmi3d_db_model!")

        # save the arguments for the database
        #self.bmi3d_query_kwargs = kwargs.pop('bmi3d_query_kwargs', dict())
        if 'bmi3d_query_kwargs' in kwargs:
            self.bmi3d_query_kwargs = kwargs['bmi3d_query_kwargs']
        else:
            self.bmi3d_query_kwargs = dict()

        super(InstanceFromDB, self).__init__(*args, **kwargs)


class DataFile(InstanceFromDB):
    def __init__(self, *args, **kwargs):
        kwargs['bmi3d_db_model'] = 'DataFile'
        super(DataFile, self).__init__(*args, **kwargs)


class OptionsList(traits.Enum):
    '''
    Wrapper around Enum so that we can keep track of the possible enumerations in list
    called 'bmi3d_input_options' which will be hidden in the UI
    '''
    def __init__(self, *args, **kwargs):
        if 'bmi3d_input_options' not in kwargs:
            kwargs['bmi3d_input_options'] = args[0]
        super(OptionsList, self).__init__(*args, **kwargs)


traits.InstanceFromDB = InstanceFromDB
traits.DataFile = DataFile
traits.OptionsList = OptionsList



from . import experiment
from . import generate
from . import report
from .experiment import Experiment, LogExperiment, Sequence, TrialTypes, FSMTable, StateTransitions

from . import task_wrapper

try:
    from .Pygame import Pygame
except:
    import warnings
    warnings.warn('riglib/experiment/__init__.py: could not import Pygame (note capital P)')
    Pygame = object

def make(exp_class, feats=(), verbose=False):
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
    if len(feats) == 0:
        return exp_class
    else:
        # construct the class list to define inheritance order for the custom task
        # inherit from the features first, then the base class
        clslist = tuple(feats) + (exp_class,)

        if verbose:
            print("metaclass constructor", clslist, feats)

        # return custom class
        return type(exp_class.__name__, clslist, dict())

