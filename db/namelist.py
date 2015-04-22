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
# try:
#     from tasklist import tasks
# except ImportError:
#     print 'create a module "tasklist" and create the task dictionaries inside it!'
#     tasks = dict()

from tasklist import tasks

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


bmi_algorithms = dict(
    KFDecoder=bmi.train.train_KFDecoder,
    PPFDecoder=bmi.train.train_PPFDecoder,
    OneDimLFPDecoder=bmi.train.create_onedimLFP,
)

bmi_training_pos_vars = [
    'cursor',
    'joint_angles',
    'plant_pos',  # used for ibmi tasks
    'mouse_state',
    'decoder_state',
]

#################################
##### State-space models for BMIs
#################################
joint_2D_state_space = bmi.state_space_models.StateSpaceNLinkPlanarChain(n_links=2, w=0.01)
tentacle_2D_state_space = bmi.state_space_models.StateSpaceNLinkPlanarChain(n_links=4, w=0.01)

from tasks.ismore_bmi_lib import StateSpaceArmAssist, StateSpaceReHand, StateSpaceIsMore
from tasks.point_mass_cursor import PointForceStateSpace
armassist_state_space = StateSpaceArmAssist()
rehand_state_space = StateSpaceReHand()
ismore_state_space = StateSpaceIsMore()

## Velocity SSMs
from riglib.bmi.state_space_models import offset_state, State
endpt_2D_states = [State('hand_px', stochastic=False, drives_obs=False, min_val=-25., max_val=25., order=0),
                   State('hand_py', stochastic=False, drives_obs=False, order=0),
                   State('hand_pz', stochastic=False, drives_obs=False, min_val=-14., max_val=14., order=0),
                   State('hand_vx', stochastic=True,  drives_obs=True, order=1),
                   State('hand_vy', stochastic=False, drives_obs=False, order=1),
                   State('hand_vz', stochastic=True,  drives_obs=True, order=1),
                   offset_state]
endpt_2D_state_space = state_space_models.LinearVelocityStateSpace(endpt_2D_states)

from tasks.speller_tasks import PointClickSSM
mouse_ssm = PointClickSSM()

bmi_state_space_models=dict(
    Endpt2D=endpt_2D_state_space,
    Tentacle=tentacle_2D_state_space,
    Armassist=armassist_state_space,
    Rehand=rehand_state_space,
    ISMORE=ismore_state_space,
    Joint2L=joint_2D_state_space,
    Mouse=mouse_ssm,
    PointMass=PointForceStateSpace(),
)

extractors = dict(
    spikecounts = bmi.extractor.BinnedSpikeCountsExtractor,
    LFPpowerMTM = bmi.extractor.LFPMTMPowerExtractor,
    LFPpowerBPF = bmi.extractor.LFPButterBPFPowerExtractor,
    EMGAmplitude = bmi.extractor.EMGAmplitudeExtractor,
)

from riglib.bmi import train
kin_extractors = dict(
    pos_vel=train.get_plant_pos_vel,
    null=train.null_kin_extractor,
)

default_extractor = "spikecounts"

bmi_update_rates = [10, 20, 30, 60, 120, 180]
