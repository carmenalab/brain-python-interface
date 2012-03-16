from riglib import experiment

features = dict(
    autostart=experiment.features.Autostart, 
    adaptive_generator=experiment.features.AdaptiveGenerator,
    button=experiment.features.Button, 
    ignore_correctness=experiment.features.IgnoreCorrectness,
    reward_system = experiment.features.RewardSystem,
    eye_data=experiment.features.EyeData,
    calibrated_eye=experiment.features.CalibratedEyeData,
    simulate_eye=experiment.features.SimulatedEyeData,
    motion_data=experiment.features.MotionData,
)

from tasks import redgreen
generators = dict(
    adaptive=experiment.generate.AdaptiveTrials,
    endless=experiment.generate.endless,
    redgreen_rand=redgreen.randcoords,

    #These are static generators
    trialtypes=experiment.generate.sequence,
    redgreen=redgreen.gencoords,
)

from tasks.rds import RDS, RDS_half
from tasks.dots import Dots
from tasks.redgreen import RedGreen, EyeCal
from tasks.button import ButtonTask
#from tasks.manualcontrol import ManualControl

tasks = dict(
    dots=Dots,
    rds=RDS,
    rds_half=RDS_half,
    redgreen=RedGreen,
    button=ButtonTask,
    eye_calibration=EyeCal,
    #manual_control=ManualControl,
)

from tracker import models
from riglib import calibrations
instance_to_model = {
    calibrations.Profile:models.Calibration,
}
