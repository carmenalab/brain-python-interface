'''
Lookup table for features, generators and tasks for experiments
'''

import numpy as np
from riglib import calibrations, bmi

from riglib.stereo_opengl.window import MatplotlibWindow
from features.generator_features import Autostart, AdaptiveGenerator, IgnoreCorrectness
from features.peripheral_device_features import Button, Joystick, DualJoystick
from features.reward_features import RewardSystem, TTLReward, JuiceLogging
from features.eyetracker_features import EyeData, CalibratedEyeData, SimulatedEyeData, FixationStart
from features.phasespace_features import MotionData, MotionSimulate, MotionAutoAlign
from features.plexon_features import PlexonBMI, RelayPlexon, RelayPlexByte
from features.blackrock_features import BlackrockBMI, RelayBlackrockByte
from features.hdf_features import SaveHDF
from features.video_recording_features import SingleChannelVideo
from features.bmi_task_features import NormFiringRates

features = dict(
    autostart=Autostart, 
    adaptive_generator=AdaptiveGenerator,
    button=Button, 
    ignore_correctness=IgnoreCorrectness,
    reward_system=RewardSystem,
    eye_data=EyeData,
    joystick=Joystick,
    dual_joystick=DualJoystick,
    calibrated_eye=CalibratedEyeData,
    eye_simulate=SimulatedEyeData,
    fixation_start=FixationStart,
    motion_data=MotionData,
    motion_simulate=MotionSimulate,
    motion_autoalign=MotionAutoAlign,
    bmi=PlexonBMI,
    blackrockbmi=BlackrockBMI,
    saveHDF=SaveHDF,
    relay_plexon=RelayPlexon,
    relay_plexbyte=RelayPlexByte,
    relay_blackrockbyte=RelayBlackrockByte,
    norm_firingrates=NormFiringRates,
    ttl_reward=TTLReward,
    juice_log=JuiceLogging,
    single_video=SingleChannelVideo,
    exp_display=MatplotlibWindow,
)

## Build the list of generators
try:
    from tasklist import tasks
except ImportError:
    print 'create a module "tasklist" and create the task dictionaries inside it!'
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
    bmi.BMI: models.Decoder,
    bmi.Decoder: models.Decoder,
} )


bmi_algorithms = dict(
    KFDecoder=bmi.train.train_KFDecoder,
    PPFDecoder=bmi.train.train_PPFDecoder,
)

bmi_training_pos_vars = ['cursor', 'joint_angles']

bmi_state_space_models=dict(
    Endpt2D=bmi.train.endpt_2D_state_space,
    Endpt3D=bmi.train.endpt_3D_state_space,
    Tentacle=bmi.train.tentacle_2D_state_space,
    Armassist=bmi.train.armassist_state_space,
    Rehand=bmi.train.rehand_state_space,
    ISMORE=bmi.train.ismore_state_space,
)

extractors = dict(
    spikecounts = bmi.extractor.BinnedSpikeCountsExtractor,
    LFPpowerMTM = bmi.extractor.LFPMTMPowerExtractor,
    LFPpowerBPF = bmi.extractor.LFPButterBPFPowerExtractor,
    EMGAmplitude = bmi.extractor.EMGAmplitudeExtractor,
)

default_extractor = "spikecounts"

bmi_update_rates = [10, 20, 30, 60, 120, 180]
