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

from tasks import generatorfunctions, redgreen, manualcontrol, sensorymapping, manualcontrolmultitasks, bmitasks, bmimultitasks, bmilowfeedback
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
    centeroutback_2D_catch=generatorfunctions.centeroutback_2D_farcatch,
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
    joystick_multi=manualcontrolmultitasks.JoystickMulti,
    joystick_move = manualcontrolmultitasks.JoystickMove,
    joystick_multi_plus_move = manualcontrolmultitasks.JoystickMulti_plusMove,
    joystickMulti_directed = manualcontrolmultitasks.JoystickMulti_Directed,
    bmi_control_multi = bmimultitasks.BMIControlMulti,
    bmi_manipulated_feedback = bmilowfeedback.BMIControlManipulatedFB,
    clda_ppf_manipulated_feedback = bmilowfeedback.CLDAControlPPFContAdaptMFB,
    clda_control_multi = bmimultitasks.CLDAControlMulti,
    sim_clda_control_multi = bmimultitasks.SimCLDAControlMulti,
    clda_rml_kf = bmimultitasks.CLDARMLKF,
    clda_cont_ppf= bmimultitasks.CLDAControlPPFContAdapt,
    test_graphics = manualcontrolmultitasks.TestGraphics,
    clda_rml_kf_ofc = bmimultitasks.CLDARMLKFOFC,
    clda_kf_cg_sb = bmimultitasks.CLDAControlKFCG,
    arm_plant = manualcontrolmultitasks.ArmPlant,
    clda_kf_cg_joint_rml = bmimultitasks.CLDAControlKFCGJoint, 
)

arms = ['RobotArm2J2D', 'RobotArm2D', 'CursorPlant', 'RobotArm5J2D']

## BMI seed tasks
# The below list shows which tasks can be used to train new Decoders
bmi_seed_tasks = ['visual_feedback_multi', 'manual_control_multi', 'joystick_multi']

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
} )


bmis = dict(
    kalman=bmi.train._train_KFDecoder_manual_control,
    kalmanCursorEpochs = bmi.train._train_KFDecoder_cursor_epochs,
    kalmanVF=bmi.train._train_KFDecoder_visual_feedback,
    kalmanVFshuf=bmi.train._train_KFDecoder_visual_feedback_shuffled,
    ppfVF=bmi.train._train_PPFDecoder_visual_feedback,
    ppfVFshuf=bmi.train._train_PPFDecoder_visual_feedback_shuffled,
    kalmanVFjoint=bmi.train._train_joint_KFDecoder_visual_feedback,
    )

