from riglib import experiment

features = dict(
    autostart=experiment.features.Autostart, 
    adaptive_generator=experiment.features.AdaptiveGenerator,
    button=experiment.features.Button, 
    ignore_correctness=experiment.features.IgnoreCorrectness,
    reward_system = experiment.features.RewardSystem,
    eye_data=experiment.features.EyeData,
    calibrated_eye=experiment.features.CalibratedEyeData,
    eye_simulate=experiment.features.SimulatedEyeData,
    fixation_start=experiment.features.FixationStart,
    motion_data=experiment.features.MotionData,
    motion_simulate=experiment.features.MotionSimulate, 
    saveHDF=experiment.features.SaveHDF
)

from tasks import redgreen
from tasks import manualcontrol
generators = dict(
    adaptive=experiment.generate.AdaptiveTrials,
    endless=experiment.generate.endless,
    redgreen_rand=redgreen.randcoords,

    #These are static generators
    trialtypes=experiment.generate.sequence,
    redgreen=redgreen.gencoords,
    reach_target=manualcontrol.rand_target_sequence,
    reach_target_2d=manualcontrol.rand_target_sequence_2d,
    reach_target_3d=manualcontrol.rand_target_sequence_3d,
)

from tasks.rds import RDS, RDS_half
from tasks.dots import Dots
from tasks.redgreen import RedGreen, EyeCal
from tasks.button import ButtonTask
from tasks.manualcontrol import FixationTraining, ManualControl, TargetCapture, MovementTraining

tasks = dict(
    dots=Dots,
    rds=RDS,
    rds_half=RDS_half,
    redgreen=RedGreen,
    button=ButtonTask,
    eye_calibration=EyeCal,
    manual_control=ManualControl,
    fixation_training=FixationTraining,
    target_capture=TargetCapture,
    movement_training=MovementTraining
)

from tracker import models
from riglib import calibrations
instance_to_model = {
    calibrations.Profile:models.Calibration,
}
