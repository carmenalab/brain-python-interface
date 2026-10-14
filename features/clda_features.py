import numpy as np
from riglib.experiment import traits
from riglib.bmi import clda

import aopy
import glob
import os


class CLDA_KFRML_IntendedVelocity(traits.HasTraits):
    clda_batch_time = traits.Float(1, desc="How frequently to update weights [s]")
    clda_update_half_life = traits.Float(60, desc="Half-life for exponential decay [s] to combine with previous weights.") #[s]
    # clda_update_batch_time = traits.Float(60, desc="How frequently to update weights [s]")
    # clda_learner_batch_time = traits.Float(60, desc="How much data to update the learner with [s]") # Samples to update intended kinematics with
    def create_learner(self):
        '''
        The "learner" uses knowledge of the task goals to determine the "intended"
        action of the BMI subject and pairs this intention estimation with actual observations.
        '''
        self.learn_flag = False
        # fmatrix = np.array(self.decoder.filt.B.T/np.max(self.decoder.filt.B))
        # self.decoder.filt.F_dict = {
        #     'target': fmatrix,
        #     'hold': np.zeros(fmatrix.shape),
        #     'timeout_penalty': np.zeros(fmatrix.shape),
        #     'wait': np.zeros(fmatrix.shape),
        #     'delay': np.zeros(fmatrix.shape),
        #     'targ_transition': np.zeros(fmatrix.shape),
        #     'hold_penalty': np.zeros(fmatrix.shape),
        #     'delay_penalty': np.zeros(fmatrix.shape),
        #     'reward': np.zeros(fmatrix.shape),
        # }
        learner_batch_size = int(self.clda_batch_time/self.decoder.binlen)
        # self.learner = clda.OFCLearnerRotateIntendedVelocity(learner_batch_size, self.decoder.filt.A, self.decoder.filt.B, self.decoder.filt.F_dict)
        self.learner = clda.SimpleEndpointIntentionLearner(learner_batch_size)

    def create_updater(self):
        '''
        The "updater" uses the output batches of data from the learner and an update rule to
        alter the decoder parameters to better match the intention estimates.
        '''
        self.updater = clda.KFRML(self.clda_batch_time, self.clda_update_half_life)
        self.updater.init(self.decoder)


class CLDA_WFSmoothbatch(traits.HasTraits):
    '''
    CLDA for Wiener filter decoders (WFDecoder), following the labgraph EMG Wiener filter CLDA:
    the intended velocity points at the target with a speed proportional to the distance from it,
    and the filter weights are periodically re-estimated by gradient descent on a regularized
    least-squares cost and blended with the previous weights (SmoothBatch).
    '''
    clda_batch_time = traits.Float(20, desc="How frequently to update weights [s]")
    clda_update_half_life = traits.Float(50, desc="Half-life for exponential decay [s] to combine with previous weights. Equal to the batch time for a 50/50 blend")
    clda_lambda_E = traits.Float(0.1, desc="Weight on the squared prediction error in the L2 cost")
    clda_lambda_D = traits.Float(10, desc="Weight on the squared norm of the filter weights (ridge penalty) in the L2 cost")
    clda_intended_velocity_gain = traits.Float(2., desc="Intended velocity = gain * (target position - cursor position) [1/s]")
    clda_solver = traits.OptionsList(("bfgs", "exact"), desc="Estimate the new weights by gradient descent (bfgs) or in closed form (exact)")
    clda_multiproc = traits.Bool(True, desc="Estimate the new weights in a separate process so the task loop is not blocked")
    clda_verbose = traits.Bool(False, desc="Print out information about the CLDA updates")

    # Task states in which the intended velocity is estimated (the target is defined and the subject
    # is trying to reach or stay on it). Covers the target capture and target tracking tasks.
    clda_learn_states = ['target', 'hold', 'delay', 'trajectory',
        'tracking_in', 'tracking_in_ramp', 'tracking_out', 'tracking_out_ramp']

    def create_learner(self):
        '''
        The "learner" pairs the intended velocity, pointing from the cursor to the target, with the observed neural features.
        '''
        self.learn_flag = False
        learner_batch_size = int(self.clda_batch_time/self.decoder.binlen)
        self.learner = clda.PositionErrorLearner(learner_batch_size, gain=self.clda_intended_velocity_gain,
            learn_states=self.clda_learn_states)

    def create_updater(self):
        '''
        The "updater" re-estimates the Wiener filter weights from each batch of data from the learner.
        '''
        self.updater = clda.WFSmoothbatch(self.clda_batch_time, self.clda_update_half_life,
            lambda_E=self.clda_lambda_E, lambda_D=self.clda_lambda_D, solver=self.clda_solver,
            verbose=self.clda_verbose, multiproc=self.clda_multiproc)
        self.updater.init(self.decoder)
    

class CLDA_Smoothbatch_IntendedVelocity(traits.HasTraits):
    clda_batch_time = traits.Float(60, desc="How frequently to update weights [s]")
    clda_half_life = traits.Float(60, desc="Half-life for exponential decay [s] to combine with previous weights.") #[s]

    def create_learner(self):
        '''
        The "learner" uses knowledge of the task goals to determine the "intended"
        action of the BMI subject and pairs this intention estimation with actual observations.
        '''
        self.learn_flag = False
        learner_batch_size = int(self.clda_batch_time/self.decoder.binlen)
        self.learner = clda.SimpleEndpointIntentionLearner(learner_batch_size)

    def create_updater(self):
        '''
        The "updater" uses the output batches of data from the learner and an update rule to
        alter the decoder parameters to better match the intention estimates.
        '''
        self.updater = clda.KFSmoothbatch(self.clda_batch_time, self.clda_half_life)
        self.updater.init(self.decoder)
