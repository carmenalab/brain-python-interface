'''
Lookup table for features, generators and tasks for experiments
'''

import numpy as np
from riglib import experiment, calibrations, bmi
from riglib.stereo_opengl.window import MatplotlibWindow

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
    bmi=experiment.features.PlexonBMI,
    blackrockbmi=experiment.features.BlackrockBMI,
    saveHDF=experiment.features.SaveHDF,
    relay_plexon=experiment.features.RelayPlexon,
    relay_plexbyte=experiment.features.RelayPlexByte,
    relay_blackrockbyte=experiment.features.RelayBlackrockByte,
    norm_firingrates=experiment.features.NormFiringRates,
    ttl_reward=experiment.features.TTLReward,
    juice_log=experiment.features.JuiceLogging,
    single_video=experiment.features.SingleChannelVideo,
    exp_display=MatplotlibWindow,
)

import tasks
from tasks import generatorfunctions, redgreen, manualcontrol, sensorymapping, manualcontrolmultitasks, bmitasks, bmimultitasks, bmilowfeedback, manualcontrolmultitasks
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
    centerout_2D_discrete_rot=generatorfunctions.centerout_2D_discrete_rot,
    centerout_2D_discrete_randorder=generatorfunctions.centerout_2D_discrete_randorder,
    centeroutback_2D_v2=generatorfunctions.centeroutback_2D,
    centeroutback_2D_catch=generatorfunctions.centeroutback_2D_farcatch,
    centeroutback_2D_catch_discrete=generatorfunctions.centeroutback_2D_farcatch_discrete,
    centerout_3D=generatorfunctions.centerout_3D,
    outcenterout_2D_discrete=generatorfunctions.outcenterout_2D_discrete,
    outcenter_2D_discrete=generatorfunctions.outcenter_2D_discrete,
    centerout_2D_discrete_offset=generatorfunctions.centerout_2D_discrete_offset,
    depth_trainer=generatorfunctions.depth_trainer,
    centerout_3D_cube=generatorfunctions.centerout_3D_cube,
    armassist_simple=generatorfunctions.armassist_simple,
    rehand_simple=generatorfunctions.rehand_simple,
    ismore_simple=generatorfunctions.ismore_simple,
    centerout_2D_discrete_multiring=generatorfunctions.centerout_2D_discrete_multiring,
    block_probabilistic_reward=generatorfunctions.colored_targets_with_probabilistic_reward,
    tentacle_multi_start_config=generatorfunctions.tentacle_multi_start_config
)


from tasks import blackrocktasks

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
    visual_feedback_multi = tasks.VisualFeedbackMulti,
    clda_auto_assist = tasks.CLDAAutoAssist,
    clda_constrained_sskf = tasks.CLDAConstrainedSSKF,
    sim_clda_control = tasks.SimCLDAControl,
    sim_bmi_control = tasks.SimBMIControl,

    ######### V2 tasks
    clda_constrained_sskf_multi = tasks.CLDAConstrainedSSKFMulti,
    manual_control_multi =tasks.ManualControlMulti,
    joystick_multi=tasks.JoystickMulti,
    joystick_leaky_vel=tasks.LeakyIntegratorVelocityJoystick,
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
    clda_kf_cg_rml = tasks.CLDAControlKFCGRML, 
    arm_plant = tasks.ArmPlant,
    clda_kf_cg_joint_rml = tasks.CLDAControlKFCGJoint,
    clda_kf_ofc_tentacle_rml = tasks.CLDAControlTentacle,
    clda_kf_ofc_tentacle_rml_base = tasks.CLDAControlTentacleBaselineReestimate,
    clda_kf_ofc_tentacle_rml_trial = tasks.CLDAControlTentacleTrialBased,
    joystick_tentacle = tasks.JoystickTentacle,
    bmi_baseline = tasks.BaselineControl,
    bmi_joint_perturb = tasks.BMIJointPerturb,
    bmi_control_tentacle_attractor = tasks.BMIControlMultiTentacleAttractor,
    bmi_cursor_bias=tasks.BMICursorBias,
    joystick_ops=tasks.JoystickDrivenCursorOPS,
    joystick_ops_bias=tasks.JoystickDrivenCursorOPSBiased,
    # joystick_freechoice=tasks.manualcontrolfreechoice.ManualControlFreeChoice,
    # joystick_freechoice_pilot = tasks.manualcontrolfreechoice.FreeChoicePilotTask,
    clda_kf_cg_rml_ivc_trial=tasks.CLDAControlKFCGRMLIVCTRIAL,
    bmi_cursor_bias_catch=bmimultitasks.BMICursorBiasCatch,
    movement_training_multi=manualcontrolmultitasks.MovementTrainingMulti,
    machine_control=bmimultitasks.TargetCaptureVisualFeedback,
    manual_control_multi_plusvar = tasks.manualcontrolmulti_COtasks.ManualControlMulti_plusvar,
    clda_tentacle_rl = tasks.CLDATentacleRL,

    ######## iBMI tasks
    ibmi_visual_feedback = blackrocktasks.VisualFeedback,
    ibmi_manual_control  = blackrocktasks.ManualControl,
    ibmi_bmi_control     = blackrocktasks.BMIControl,
    ibmi_clda_control    = blackrocktasks.CLDAControl,
    passive_exo          = tasks.RecordEncoderData,
)

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


bmi_algorithms = dict(
    KFDecoder=bmi.train.train_KFDecoder,
    PPFDecoder=bmi.train.train_PPFDecoder,
)

bmi_training_pos_vars = ['cursor', 'joint_angles']

bmi_state_space_models=dict(
    Endpt2D=bmi.train.endpt_2D_state_space,
    Endpt3D=bmi.train.endpt_3D_state_space,
    Tentacle=bmi.train.tentacle_2D_state_space,
    Armassist=bmi.train.armassist_state_space,
    Rehand=bmi.train.rehand_state_space,
    ISMORE=bmi.train.ismore_state_space,
)

extractors = dict(
    spikecounts = bmi.extractor.BinnedSpikeCountsExtractor,
    LFPpowerMTM = bmi.extractor.LFPMTMPowerExtractor,
    LFPpowerBPF = bmi.extractor.LFPButterBPFPowerExtractor,
    EMGAmplitude = bmi.extractor.EMGAmplitudeExtractor,
)

default_extractor = "spikecounts"

bmi_update_rates = [10, 20, 30, 60, 120, 180]
