from riglib import experiment
from riglib import calibrations, bmi

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
    motion_autoalign=experiment.features.MotionAutoAlign,
    bmi=experiment.features.SpikeBMI,
    saveHDF=experiment.features.SaveHDF,
    relay_plexon=experiment.features.RelayPlexon,
    relay_plexbyte=experiment.features.RelayPlexByte,
)

from tasks import redgreen
from tasks import manualcontrol
from tasks import sensorymapping
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
    nummap=sensorymapping.gen_taps,
)

from tasks.rds import RDS, RDS_half
from tasks.dots import Dots
from tasks.redgreen import RedGreen, EyeCal
from tasks.button import ButtonTask
from tasks.manualcontrol import FixationTraining, ManualControl, TargetCapture, MovementTraining, TargetDirection, TestBoundary, BMIControl
from tasks.sensorymapping import FreeMap, ArmPositionTraining, NumberMap

tasks = dict(
    dots=Dots,
    rds=RDS,
    rds_half=RDS_half,
    redgreen=RedGreen,
    button=ButtonTask,
    eye_calibration=EyeCal,
    manual_control=ManualControl,
    bmi_control=BMIControl,
    fixation_training=FixationTraining,
    target_capture=TargetCapture,
    movement_training=MovementTraining,
    direction_training=TargetDirection,
    test_boundary=TestBoundary,
    free_map=FreeMap,
    arm_position_training=ArmPositionTraining,
    number_map=NumberMap,
)

from tracker import models

class SubclassDict(dict):
    '''A special dict that returns the associated model if the queried item is a subclass of any of the keys'''
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
} )


bmis = dict(
    velocity_kalman= bmi.KalmanFilter,
    )