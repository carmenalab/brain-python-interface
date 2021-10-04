'''
Module for the core "features" that can be used to extend and customize a 
task/experiment by multiple inheritance.
'''
from riglib.stereo_opengl.window import WindowWithExperimenterDisplay, Window2D
from .generator_features import Autostart, AdaptiveGenerator, IgnoreCorrectness
from .peripheral_device_features import Button, Joystick, DualJoystick, Joystick_plus_TouchSensor, KeyboardControl, MouseControl
from .reward_features import RewardSystem, TTLReward, JuiceLogging
from .eyetracker_features import EyeData, CalibratedEyeData, SimulatedEyeData, FixationStart
from .phasespace_features import MotionData, MotionSimulate, MotionAutoAlign
from .optitrack_features import Optitrack
from .plexon_features import PlexonBMI, RelayPlexon, RelayPlexByte
from .hdf_features import SaveHDF
from .video_recording_features import SingleChannelVideo
from .bmi_task_features import NormFiringRates
from .arduino_features import PlexonSerialDIORowByte
from .blackrock_features import BlackrockBMI
from .blackrock_features import RelayBlackrockByte
from .ecube_features import EcubeFileBMI, EcubeBMI

built_in_features = dict(
    keyboard=KeyboardControl,
    mouse=MouseControl,
    optitrack=Optitrack,
    reward_system=RewardSystem,
    saveHDF=SaveHDF,
    autostart=Autostart,
    window2D=Window2D,
    adaptive_generator=AdaptiveGenerator,
    button=Button,
    ignore_correctness=IgnoreCorrectness,
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
    norm_firingrates=NormFiringRates,
    ttl_reward=TTLReward,
    juice_log=JuiceLogging,
    single_video=SingleChannelVideo,
    exp_display=WindowWithExperimenterDisplay,
    relay_arduino=PlexonSerialDIORowByte,
    plexonbmi=PlexonBMI,
    relay_plexon=RelayPlexon,
    relay_plexbyte=RelayPlexByte,
    blackrockbmi        = BlackrockBMI,
    relay_blackrockbyte = RelayBlackrockByte,
    ecube_playback_bmi = EcubeFileBMI,
    ecube_bmi = EcubeBMI,
)

# >>> features.built_in_features['autostart'].__module__
# 'features.generator_features'
# >>> features.built_in_features['autostart'].__qualname__
# 'Autostart'