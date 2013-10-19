'''
Lookup table for features, generators and tasks for experiments
'''


from riglib import experiment, calibrations, bmi

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

from tasks import generatorfunctions, redgreen, manualcontrol, sensorymapping, manualcontrolmultitasks, bmitasks, bmimultitasks
generators = dict(
    adaptive=experiment.generate.AdaptiveTrials,
    endless=experiment.generate.endless,
    redgreen_rand=redgreen.randcoords,

    #These are static generators
    trialtypes=experiment.generate.sequence,
    redgreen=redgreen.gencoords,
    reach_target_2d=manualcontrol.rand_target_sequence_2d,
    reach_target_3d=manualcontrol.rand_target_sequence_3d,
    centerout_2d=manualcontrol.rand_target_sequence_2d_centerout,
    nummap=sensorymapping.gen_taps,
    centerout_partial=manualcontrol.rand_target_sequence_2d_partial_centerout,
    centerout_back=manualcontrol.rand_multi_sequence_2d_centeroutback,
    centerout_2step=manualcontrol.rand_multi_sequence_2d_centerout2step,
    centerout_2D_discrete=generatorfunctions.centerout_2D_discrete,
    centeroutback_2D_v2=generatorfunctions.centeroutback_2D,
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
    manual_control=manualcontrol.ManualControl,
    bmi_control=bmitasks.BMIControl,
    clda_control=bmitasks.CLDAControl,
    manual_predict=bmitasks.ManualWithPredictions,
    fixation_training=manualcontrol.FixationTraining,
    target_capture=manualcontrol.TargetCapture,
    movement_training=manualcontrol.MovementTraining,
    direction_training=manualcontrol.TargetDirection,
    test_boundary=manualcontrol.TestBoundary,
    free_map=sensorymapping.FreeMap,
    arm_position_training=sensorymapping.ArmPositionTraining,
    number_map=sensorymapping.NumberMap,
    joystick_control = manualcontrol.JoystickControl,
    manual_control_2 = manualcontrol.ManualControl2,
    visual_feedback = bmitasks.VisualFeedback,
    visual_feedback_multi = bmimultitasks.VisualFeedbackMulti,
    clda_auto_assist = bmitasks.CLDAAutoAssist,
    clda_constrained_sskf = bmitasks.CLDAConstrainedSSKF,
    sim_clda_control = bmitasks.SimCLDAControl,
    sim_bmi_control = bmitasks.SimBMIControl,

    ######### V2 tasks
    clda_constrained_sskf_multi = bmimultitasks.CLDAConstrainedSSKFMulti,
    manual_control_multi =manualcontrolmultitasks.ManualControlMulti,
    bmi_control_multi = bmimultitasks.BMIControlMulti,
    clda_control_multi = bmimultitasks.CLDAControlMulti,
    clda_rml_kf = bmimultitasks.CLDARMLKF,
    test_graphics = manualcontrolmultitasks.TestGraphics,
    two_link_arm = manualcontrolmultitasks.TwoLinkArm,
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
