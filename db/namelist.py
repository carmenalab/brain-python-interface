'''Needs docs'''


from riglib import experiment
from riglib import calibrations, bmi

features = dict(
    autostart=experiment.features.Autostart, 
    adaptive_generator=experiment.features.AdaptiveGenerator,
    button=experiment.features.Button, 
    ignore_correctness=experiment.features.IgnoreCorrectness,
    reward_system = experiment.features.RewardSystem,
    eye_data=experiment.features.EyeData,
    joystick=experiment.features.Joystick,
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
    norm_firingrates=experiment.features.NormFiringRates,
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
    #reach_target=manualcontrol.rand_target_sequence,
    reach_target_2d=manualcontrol.rand_target_sequence_2d,
    reach_target_3d=manualcontrol.rand_target_sequence_3d,
    centerout_2d=manualcontrol.rand_target_sequence_2d_centerout,
    nummap=sensorymapping.gen_taps,
    centerout_partial=manualcontrol.rand_target_sequence_2d_partial_centerout,
    centerout_back=manualcontrol.rand_multi_sequence_2d_centeroutback,
    centerout_2step=manualcontrol.rand_multi_sequence_2d_centerout2step,
)

from tasks.rds import RDS, RDS_half
from tasks.dots import Dots
from tasks.redgreen import RedGreen, EyeCal
from tasks.button import ButtonTask
import tasks.manualcontrol #import FixationTraining, ManualControl, ManualControl2, TargetCapture, MovementTraining, JoystickControl, TargetDirection, TestBoundary
import tasks.sensorymapping #import FreeMap, ArmPositionTraining, NumberMap
import tasks.bmitasks #import BMIControl, ManualWithPredictions, CLDAControl

tasks = dict(
    dots=Dots,
    rds=RDS,
    rds_half=RDS_half,
    redgreen=RedGreen,
    button=ButtonTask,
    eye_calibration=EyeCal,
    manual_control=tasks.manualcontrol.ManualControl,
    bmi_control=tasks.bmitasks.BMIControl,
    clda_control=tasks.bmitasks.CLDAControl,
    manual_predict=tasks.bmitasks.ManualWithPredictions,
    fixation_training=tasks.manualcontrol.FixationTraining,
    target_capture=tasks.manualcontrol.TargetCapture,
    movement_training=tasks.manualcontrol.MovementTraining,
    direction_training=tasks.manualcontrol.TargetDirection,
    test_boundary=tasks.manualcontrol.TestBoundary,
    free_map=tasks.sensorymapping.FreeMap,
    arm_position_training=tasks.sensorymapping.ArmPositionTraining,
    number_map=tasks.sensorymapping.NumberMap,
    joystick_control = tasks.manualcontrol.JoystickControl,
    joystick_targ_direc = tasks.manualcontrol.CorrectTargetDir,
    manual_control_2 = tasks.manualcontrol.ManualControl2,
    visual_feedback = tasks.bmitasks.VisualFeedback,
    clda_auto_assist = tasks.bmitasks.CLDAAutoAssist,
    clda_constrained_sskf = tasks.bmitasks.CLDAConstrainedSSKF,
    sim_clda_control = tasks.bmitasks.SimCLDAControl,
    sim_bmi_control = tasks.bmitasks.SimBMIControl,

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
    kalman=bmi.train._train_KFDecoder_manual_control,
    kalmanVF=bmi.train._train_KFDecoder_visual_feedback,
    )
