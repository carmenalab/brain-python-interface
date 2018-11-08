'''
Lookup table for features, generators and tasks for experiments
'''

## Get the list of experiment features
from .featurelist import features

## Get the list of tasks
from .tasklist import tasks


# Derive generator functions from the tasklist (all generatorfunctions should be staticmethods of a task)
generator_names = []
generator_functions = []
for task in tasks:
    task_cls = tasks[task]
    if hasattr(task_cls, 'sequence_generators'):
        generator_function_names = task_cls.sequence_generators
        gen_fns = [getattr(task_cls, x) for x in generator_function_names]
        for fn_name, fn in zip(generator_function_names, gen_fns):
            if fn in generator_functions:
                pass
            else:
                generator_names.append(fn_name)
                generator_functions.append(fn)

generators = dict()
for fn_name, fn in zip(generator_names, generator_functions):
    generators[fn_name] = fn


################################################################################
################################################################################
class SubclassDict(dict):
    '''
    A special dict that returns the associated Django database model 
    if the queried item is a subclass of any of the keys
    '''
    def __getitem__(self, name):
        try:
            return super(self.__class__, self).__getitem__(name)
        except KeyError:
            for inst, model in list(self.items()):
                if issubclass(name, inst):
                    return model
        raise KeyError

# from riglib.plants import RefTrajectories
# from ismore.emg_decoding import LinearEMGDecoder
# instance_to_model = SubclassDict( {
#     calibrations.Profile:models.Calibration,
#     calibrations.AutoAlign:models.AutoAlignment,
#     BMI: models.Decoder,
#     Decoder: models.Decoder,
#     RefTrajectories: models.DataFile,
#     LinearEMGDecoder: models.Decoder,
# } )

# instance_to_model_filter_kwargs = SubclassDict( {
#     calibrations.Profile:dict(),
#     calibrations.AutoAlign:dict(),
#     BMI:dict(),
#     Decoder:dict(),
#     RefTrajectories: dict(system__name='ref_trajectories'),
#     LinearEMGDecoder: dict(name__startswith='emg_decoder')
# } )

################################################################################
################################################################################
from .bmilist import *