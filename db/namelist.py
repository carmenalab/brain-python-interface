'''
Lookup table for features, generators and tasks for experiments
'''

import numpy as np
from riglib import calibrations, bmi
from riglib.bmi.bmi import BMI, Decoder
from riglib.bmi import state_space_models

## Get the list of experiment features
try:
    from featurelist import features
except:
    features = dict()

## Get the list of tasks
try:
    from tasklist import tasks
except ImportError:
    tasks = dict()

from itertools import izip

# Derive generator functions from the tasklist (all generatorfunctions should be staticmethods of a task)
generator_names = []
generator_functions = []
for task in tasks:
    task_cls = tasks[task]
    generator_function_names = task_cls.sequence_generators
    gen_fns = [getattr(task_cls, x) for x in generator_function_names]
    for fn_name, fn in izip(generator_function_names, gen_fns):
        if fn in generator_functions:
            pass
        else:
            generator_names.append(fn_name)
            generator_functions.append(fn)

generators = dict()
for fn_name, fn in izip(generator_names, generator_functions):
    generators[fn_name] = fn

from tracker import models
class SubclassDict(dict):
    '''
    A special dict that returns the associated Django database model 
    if the queried item is a subclass of any of the keys
    '''
    def __getitem__(self, name):
        try:
            return super(self.__class__, self).__getitem__(name)
        except KeyError:
            for inst, model in self.items():
                if issubclass(name, inst):
                    return model
        raise KeyError
        
instance_to_model = SubclassDict( {
    calibrations.Profile:models.Calibration,
    calibrations.AutoAlign:models.AutoAlignment,
    BMI: models.Decoder,
    Decoder: models.Decoder,
} )

try:
    from bmilist import bmi_algorithms
    from bmilist import bmi_training_pos_vars
    from bmilist import bmi_state_space_models
    from bmilist import extractors
    from bmilist import default_extractor
    from bmilist import bmi_update_rates
except:
    import "error importing BMI configuration variables"
    import traceback
    traceback.print_exc()
