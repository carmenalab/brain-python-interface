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
    dual_joystick=experiment.features.DualJoystick,
    calibrated_eye=experiment.features.CalibratedEyeData,
    eye_simulate=experiment.features.SimulatedEyeData,
    fixation_start=experiment.features.FixationStart,
    motion_data=experiment.features.MotionData,
    motion_simulate=experiment.features.MotionSimulate, 
    motion_autoalign=experiment.features.MotionAutoAlign,
    bmi=experiment.features.SpikeBMI,
    blackrockbmi=experiment.features.BlackrockBMI,
    saveHDF=experiment.features.SaveHDF,
    relay_plexon=experiment.features.RelayPlexon,
    relay_plexbyte=experiment.features.RelayPlexByte,
    norm_firingrates=experiment.features.NormFiringRates,
    lfpbmi=None,
    continous_bmi=None,
)

import tasks
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
    centeroutback_2D_catch_discrete=generatorfunctions.centeroutback_2D_farcatch_discrete,
    centerout_3D=generatorfunctions.centerout_3D,
    outcenterout_2D_discrete=generatorfunctions.outcenterout_2D_discrete,
    outcenter_2D_discrete=generatorfunctions.outcenter_2D_discrete,
)

# from tasks.rds import RDS, RDS_half
# from tasks.dots import Dots
# from tasks.redgreen import RedGreen, EyeCal
# from tasks.button import ButtonTask

tasks = dict(
    dots=tasks.Dots,
    rds=tasks.RDS,
    rds_half=tasks.RDS_half,
    redgreen=tasks.RedGreen,
    button=tasks.ButtonTask,
    eye_calibration=tasks.EyeCal,
    manual_control=tasks.ManualControl,
    bmi_control=tasks.BMIControl,
    clda_control=tasks.CLDAControl,
    manual_predict=tasks.ManualWithPredictions,
    fixation_training=tasks.FixationTraining,
    target_capture=tasks.TargetCapture,
    movement_training=tasks.MovementTraining,
    direction_training=tasks.TargetDirection,
    test_boundary=tasks.TestBoundary,
    free_map=tasks.FreeMap,
    arm_position_training=tasks.ArmPositionTraining,
    number_map=tasks.NumberMap,
    joystick_control = tasks.JoystickControl,
    manual_control_2 = tasks.ManualControl2,
    visual_feedback = tasks.VisualFeedback,
    # visual_feedback_multi = tasks.VisualFeedbackMulti,
    visual_feedback_multi = tasks.blackrocktasks.VisualFeedbackMulti,
    clda_auto_assist = tasks.CLDAAutoAssist,
    clda_constrained_sskf = tasks.CLDAConstrainedSSKF,
    sim_clda_control = tasks.SimCLDAControl,
    sim_bmi_control = tasks.SimBMIControl,

    ######### V2 tasks
    clda_constrained_sskf_multi = tasks.CLDAConstrainedSSKFMulti,
    manual_control_multi =tasks.ManualControlMulti,
    joystick_multi=tasks.JoystickMulti,
    joystick_move = tasks.JoystickMove,
    joystick_multi_plus_move = tasks.JoystickMulti_plusMove,
    joystick_multi_directed = tasks.JoystickMulti_Directed,
    bmi_control_multi = tasks.BMIControlMulti,
    bmi_manipulated_feedback = tasks.BMIControlManipulatedFB,
    clda_ppf_manipulated_feedback = tasks.CLDAControlPPFContAdaptMFB,
    clda_control_multi = tasks.CLDAControlMulti,
    sim_clda_control_multi = tasks.SimCLDAControlMulti,
    clda_rml_kf = tasks.CLDARMLKF,
    clda_cont_ppf= tasks.CLDAControlPPFContAdapt,
    test_graphics = tasks.TestGraphics,
    clda_rml_kf_ofc = tasks.CLDARMLKFOFC,
    clda_kf_cg_sb = tasks.CLDAControlKFCG,
    arm_plant = tasks.ArmPlant,
    clda_kf_cg_joint_rml = tasks.CLDAControlKFCGJoint,
    clda_kf_ofc_tentacle_rml = tasks.CLDAControlTentacle,
    clda_kf_ofc_tentacle_rml_base = tasks.CLDAControlTentacleBaselineReestimate,
    clda_kf_ofc_tentacle_rml_trial = tasks.CLDAControlTentacleTrialBased,
    joystick_tentacle = tasks.JoystickTentacle,
    bmi_baseline = tasks.BaselineControl,
    bmi_joint_perturb = tasks.BMIJointPerturb,
    bmi_control_tentacle_attractor = tasks.BMIControlMultiTentacleAttractor,
)

arms = ['RobotArm2J2D', 'RobotArm2D', 'CursorPlant', 'RobotArmGen2D']

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
    bmi.Decoder: models.Decoder,
} )


bmis = dict(
    kalman=bmi.train._train_KFDecoder_manual_control,
    kalmanCursorEpochs = bmi.train._train_KFDecoder_cursor_epochs,
    kalmanVF=bmi.train._train_KFDecoder_visual_feedback,
    kalmanVFshuf=bmi.train._train_KFDecoder_visual_feedback_shuffled,
    ppfVF=bmi.train._train_PPFDecoder_visual_feedback,
    ppfVFshuf=bmi.train._train_PPFDecoder_visual_feedback_shuffled,
    kalmanVFjoint=bmi.train._train_joint_KFDecoder_visual_feedback,
    kalmanVFtentacle=bmi.train._train_tentacle_KFDecoder_visual_feedback,
    kalmanVFarmassist=bmi.train._train_armassist_KFDecoder_visual_feedback,
)

extractors = dict(
    spikecounts = bmi.extractor.BinnedSpikeCountsExtractor,
    LFPpowerMTM = bmi.extractor.LFPMTMPowerExtractor,
    LFPpowerBPF = bmi.extractor.LFPButterBPFPowerExtractor,
    EMGAmplitude = bmi.extractor.EMGAmplitudeExtractor,
)

default_extractor = "spikecounts"

