from riglib import experiment

features = dict(
    autostart=experiment.features.Autostart, 
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

tasks = dict(
    dots=Dots,
    rds=RDS,
    rds_half=RDS_half,
    redgreen=RedGreen,
    button=ButtonTask,
    eye_calibration=EyeCal,
)

from tracker import models
from riglib import calibrations
instance_to_model = {
    calibrations.Profile:models.Calibration,
}