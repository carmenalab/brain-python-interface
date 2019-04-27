'''
Module for the core "features" that can be used to extend and customize a 
task/experiment by multiple inheritance.
''' 

from riglib.stereo_opengl.window import WindowWithExperimenterDisplay
from .generator_features import Autostart, AdaptiveGenerator, IgnoreCorrectness
from .peripheral_device_features import Button, Joystick, DualJoystick, Joystick_plus_TouchSensor
from .reward_features import RewardSystem, TTLReward, JuiceLogging
from .eyetracker_features import EyeData, CalibratedEyeData, SimulatedEyeData, FixationStart
from .phasespace_features import MotionData, MotionSimulate, MotionAutoAlign
from .plexon_features import PlexonBMI, RelayPlexon, RelayPlexByte
from .hdf_features import SaveHDF
from .video_recording_features import SingleChannelVideo
from .bmi_task_features import NormFiringRates
from .arduino_features import PlexonSerialDIORowByte

built_in_features = dict(
    autostart=Autostart,
    adaptive_generator=AdaptiveGenerator,
    button=Button,
    ignore_correctness=IgnoreCorrectness,
    reward_system=RewardSystem,
    eye_data=EyeData,
    joystick=Joystick,
    dual_joystick=DualJoystick,
    joystick_and_touch = Joystick_plus_TouchSensor,
    calibrated_eye=CalibratedEyeData,
    eye_simulate=SimulatedEyeData,
    fixation_start=FixationStart,
    motion_data=MotionData,
    motion_simulate=MotionSimulate,
    motion_autoalign=MotionAutoAlign,
    bmi=PlexonBMI,
    saveHDF=SaveHDF,
    relay_plexon=RelayPlexon,
    relay_plexbyte=RelayPlexByte,
    norm_firingrates=NormFiringRates,
    ttl_reward=TTLReward,
    juice_log=JuiceLogging,
    single_video=SingleChannelVideo,
    exp_display=WindowWithExperimenterDisplay,
    relay_arduino=PlexonSerialDIORowByte,
)

# >>> features.built_in_features['autostart'].__module__
# 'features.generator_features'
# >>> features.built_in_features['autostart'].__qualname__
# 'Autostart'