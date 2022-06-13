import numpy as np
from riglib import bmi
from riglib.bmi import state_space_models, train, extractor

bmi_update_rates = [10, 20, 30, 60, 120, 180]

bmi_algorithms = dict(
    KFDecoder=bmi.train.train_KFDecoder,
    PPFDecoder=bmi.train.train_PPFDecoder,
    OneDimLFPDecoder=bmi.train.create_onedimLFP,
    LinearDecoder=bmi.train.create_lindecoder,
)

bmi_training_pos_vars = [
    'cursor',
    'joint_angles',
    'plant_pos',  # used for ibmi tasks
    'mouse_state',
    'decoder_state',
]

## State-space models for BMIs
joint_2D_state_space = bmi.state_space_models.StateSpaceNLinkPlanarChain(n_links=2, w=0.01)
tentacle_2D_state_space = bmi.state_space_models.StateSpaceNLinkPlanarChain(n_links=4, w=0.01)
endpt_2D_state_space = bmi.state_space_models.StateSpaceEndptVel2D()

# ## Velocity SSMs
# from riglib.bmi.state_space_models import offset_state, State
# endpt_2D_states = [State('hand_px', stochastic=False, drives_obs=False, min_val=-25., max_val=25., order=0),
#                    State('hand_py', stochastic=False, drives_obs=False, order=0),
#                    State('hand_pz', stochastic=False, drives_obs=False, min_val=-14., max_val=14., order=0),
#                    State('hand_vx', stochastic=True,  drives_obs=True, order=1),
#                    State('hand_vy', stochastic=False, drives_obs=False, order=1),
#                    State('hand_vz', stochastic=True,  drives_obs=True, order=1),
#                    offset_state]
# endpt_2D_state_space = state_space_models.LinearVelocityStateSpace(endpt_2D_states)

bmi_state_space_models=dict(
    Endpt2D=endpt_2D_state_space,
    Tentacle=tentacle_2D_state_space,
    Joint2L=joint_2D_state_space,
)

extractors = dict(
    spikecounts = extractor.BinnedSpikeCountsExtractor,
    LFPpowerMTM = extractor.LFPMTMPowerExtractor,
    direct = extractor.DirectObsExtractor,
)

kin_extractors = dict(
    pos_vel=train.get_plant_pos_vel,
    null=train.null_kin_extractor,
)

default_extractor = "spikecounts"

