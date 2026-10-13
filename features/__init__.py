'''
Module for the core "features" that can be used to extend and customize a 
task/experiment by multiple inheritance.
'''

from features.debug_features import Profiler, OnlineAnalysis
from features.laser_features import QwalorLaser, MultiQwalorLaser, SwitchedQwalorLaser, LaserState
from features.quattrocento_features import QuattBMI
from riglib.stereo_opengl.window import WindowWithExperimenterDisplay, Window2D, Window2DIn3D
from riglib.stereo_opengl.openxr import WindowVR

from .generator_features import Autostart, RandomDelay, AdaptiveGenerator, IgnoreCorrectness, PoissonWait, Progressbar_fixation, \
    HideLeftTrajectory, StartTrialBelowSpeedThr, ReadysetMedley, ReadysetColorChange, DiscreteRandomDelay_EyeHandSequence, HideCursorReturn
from .peripheral_device_features import Button, Joystick, DualJoystick, Joystick_plus_TouchSensor, KeyboardControl, MouseControl, ForceControl
from .reward_features import RewardSystem, RewardSystemPulse, TTLReward, JuiceLogging, PelletReward, JackpotRewards, ProgressBar, TrackingRewards, RewardAudio, PenaltyAudio, ScoreRewards, ConsecutiveJackpot
from .eyetracker_features import EyeCursor, EyeData, CalibratedEyeData, PupilLabStreaming, SimulatedEyeData, FixationStart, EyeConstrained, \
    AutomaticEyeCalibration, EyeCalibration, EyeStreaming
from .touch_features import MouseEmulateTouch, TabletTouch
from .phasespace_features import MotionData, MotionSimulate, MotionAutoAlign
from .optitrack_features import Optitrack, SpheresToCylinders
from .plexon_features import PlexonBMI, RelayPlexon, RelayPlexByte
from .spikerbox_features import SpikerBoxBMI
from .hdf_features import SaveHDF
from .video_recording_features import SingleChannelVideo, E3Video
from .bmi_task_features import NormFiringRates
from .arduino_features import PlexonSerialDIORowByte
from .blackrock_features import BlackrockBMI
from .blackrock_features import RelayBlackrockByte
from .ecube_features import EcubeFileBMI, EcubeBMI, RecordECube
from .sync_features import ArduinoSync, CursorAnalogOut, ScreenSync, HDFSync
from .neuropixel_features import RecordNeuropixels
from .clda_features import CLDA_Smoothbatch_IntendedVelocity, CLDA_KFRML_IntendedVelocity, CLDA_WFSmoothbatch

built_in_features = dict(
    keyboard=KeyboardControl,
    mouse=MouseControl,
    optitrack=Optitrack,
    reward_system=RewardSystem, 
    pellet_reward=PelletReward,
    reward_audio=RewardAudio,
    penalty_audio=PenaltyAudio,
    reward_score=ScoreRewards,
    saveHDF=SaveHDF,
    syncHDF=HDFSync,
    autostart=Autostart,
    rand_delay=RandomDelay,
    poisson_wait=PoissonWait,
    window2D=Window2D,
    window2Din3D=Window2DIn3D,
    windowVR=WindowVR,
    cylinder_center_out=SpheresToCylinders,
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
    jackpot_rewards=JackpotRewards,
    ttl_reward=TTLReward,
    juice_log=JuiceLogging,
    single_video=SingleChannelVideo,
    exp_display=WindowWithExperimenterDisplay,
    relay_arduino=PlexonSerialDIORowByte,
    spikerbox_bmi=SpikerBoxBMI,
    plexonbmi=PlexonBMI,
    relay_plexon=RelayPlexon,
    relay_plexbyte=RelayPlexByte,
    blackrockbmi        = BlackrockBMI,
    relay_blackrockbyte = RelayBlackrockByte,
    ecube_playback_bmi = EcubeFileBMI,
    ecube_bmi = EcubeBMI,
    ecube_start = RecordECube,
    quatt_bmi = QuattBMI,
    qwalor_laser = QwalorLaser,
    multi_qwalor_laser = MultiQwalorLaser,
    switched_qwalor_laser = SwitchedQwalorLaser,
    laser_state = LaserState,
    e3video = E3Video,
    debug = Profiler,
    online_analysis = OnlineAnalysis,
    arduino_sync=ArduinoSync,
    screen_sync=ScreenSync,
    cursor_sync=CursorAnalogOut,
    progress_bar=ProgressBar,
    tracking_rewards=TrackingRewards,
    neuropixel=RecordNeuropixels,
    eye_streaming=EyeStreaming,
    pupillab_streaming=PupilLabStreaming,
    eye_cursor=EyeCursor,
    eye_constrained=EyeConstrained,
    eye_calibration=EyeCalibration, 
    force_sensor=ForceControl,
    show_fixation_progress=Progressbar_fixation,
    clda_smoothbatch=CLDA_Smoothbatch_IntendedVelocity,
    clda_kfrml=CLDA_KFRML_IntendedVelocity,
    clda_wf_smoothbatch=CLDA_WFSmoothbatch,
    hide_left_trajectory=HideLeftTrajectory,
    consecutive_jackpot=ConsecutiveJackpot,
    mouse_emulate_touch=MouseEmulateTouch,
    readyset_medley=ReadysetMedley,
    start_trial_below_speed_thr=StartTrialBelowSpeedThr,
    reward_system_pulse = RewardSystemPulse,
    readyset_colorchange = ReadysetColorChange,
    tablet_touch = TabletTouch,
    random_delay_eye_hand_sequence = DiscreteRandomDelay_EyeHandSequence,
    hide_cursor_return = HideCursorReturn,
    auto_eye_calibration = AutomaticEyeCalibration,
)

# >>> features.built_in_features['autostart'].__module__
# 'features.generator_features'
# >>> features.built_in_features['autostart'].__qualname__
# 'Autostart'