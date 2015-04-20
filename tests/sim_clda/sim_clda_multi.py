#!/usr/bin/python
"""
Simulation of CLDA control task
"""
## Imports
from __future__ import division
from db import dbfunctions
from db.tracker import dbq
from db.tracker import models

import os
import numpy as np
import multiprocessing as mp
from scipy.io import loadmat, savemat

from features.generator_features import Autostart
from features.simulation_features import SimHDF, SimTime, SimCosineTunedPointProc, SimPPFDecoderCursor, SimPPFDecoderCursorShuffled
from features.hdf_features import SaveHDF

import riglib.bmi
from riglib.bmi import train, kfdecoder, clda, ppfdecoder
from tasks import bmimultitasks, generatorfunctions as genfns
from riglib.stereo_opengl.window import WindowDispl2D
from tasks import cursor_clda_tasks

from riglib.bmi.feedback_controllers import LQRController

import pickle


reload(kfdecoder)
reload(ppfdecoder)
reload(clda)
reload(riglib.bmi)
reload(riglib.bmi.train)

import argparse
parser = argparse.ArgumentParser(description='Analyze neural control of a redundant kinematic chain')
parser.add_argument('--clean', help='', action="store_true")
parser.add_argument('--show', help='', action="store_true")
parser.add_argument('--alg', help='', action="store")
parser.add_argument('--save', help='', action="store_true")

args = parser.parse_args()

#######################
#### RML test case ####
#######################
class SimRML(SaveHDF, Autostart, SimTime, WindowDispl2D, cursor_clda_tasks.SimCLDAControlMulti):
    update_rate = 0.1
    starting_pos = (0., 0., 0.)
    rand_start = (0., 0.)    
    assist_level = (0., 0.)
    target_radius = 1.8
    plant_type = 'cursor_14x14'
    win_res = (250, 140)
    def __init__(self, *args, **kwargs):
        super(SimRML, self).__init__(*args, **kwargs)
        self.batch_time = 0.1
        self.half_life  = (20.0, 20.0)
        self.starting_pos = (0., 0., 0.)
        self.assist_time = 15.

    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])

class PointProcNeuralSim(object):
    def _init_neural_encoder(self):
        from riglib.bmi import sim_neurons
        sim_encoder_fname = os.path.join(os.getenv('HOME'), 'code/bmi3d/tests/ppf', 'sample_spikes_and_kinematics_10000.mat')
        self.encoder = sim_neurons.load_ppf_encoder_2D_vel_tuning_clda_sim(sim_encoder_fname, dt=1./60)

class PointProcSmoothbatch(object):
    def create_updater(self):
        dt = 1./180
        batch_time = 1./60
        batch_size = batch_time/dt
        half_life = 120.
        rho = np.exp(np.log(0.5) / (half_life/batch_time))
        
        #self.updater = clda.PPFContinuousBayesianUpdater(self.decoder)
        self.updater = clda.PPFSmoothbatchSingleThread()
        self.updater.rho = rho
        self.updater.batch_time = batch_time

class PointProcContinuous(object):
    def create_updater(self):
        self.updater = clda.PPFContinuousBayesianUpdater(self.decoder)

 

from tasks.cursor_clda_tasks import OFCLearner3DEndptPPF
from riglib.experiment import traits
from riglib.bmi import feedback_controllers

# OFCLearner3DEndptPPF(1)
class SimCLDAControlMultiDispl2D_PPF(SimTime, Autostart, WindowDispl2D, SimCosineTunedPointProc, SimPPFDecoderCursorShuffled, cursor_clda_tasks.CLDAControlPPFContAdapt2):
# class SimCLDAControlMultiDispl2D_PPF(Autostart, WindowDispl2D, SimCosineTunedPointProc, SimPPFDecoderCursor, cursor_clda_tasks.CLDAControlPPFContAdapt2):
    win_res = (250, 140)
    tau = traits.Float(2.7, desc="Magic parameter for speed of OFC.")
    param_noise_scale = traits.Float(1.0, desc="Stuff")
    half_life = (0, 0)
    half_life_time = 1

    def __init__(self, *args, **kwargs):
        from riglib.bmi.state_space_models import StateSpaceEndptVel2D
        ssm = StateSpaceEndptVel2D()
        A, B, W = ssm.get_ssm_matrices(update_rate=1./180)
        Q = np.mat(np.diag([1., 1, 1, 0, 0, 0, 0]))
        R = 100*np.mat(np.diag([1., 1., 1.]))
        self.fb_ctrl = LQRController(A, B, Q, R)

        self.ssm = ssm


        super(SimCLDAControlMultiDispl2D_PPF, self).__init__(*args, **kwargs)
        self.batch_time = 1./10 #60.  # TODO 10 Hz running seems to be hardcoded somewhere
        self.assist_level = 0., 0.
        self.assist_level_time = 60.
        self.last_get_spike_counts_time = -1./60
        self.learn_flag = True

    def get_features(self):
        feats = super(SimCLDAControlMultiDispl2D_PPF, self).get_features()
        # if np.any(feats['spike_counts']):
        #     print "spikes!"
        #     print feats
        # print np.sum(feats['spike_counts'], axis=0)
        return feats




    def create_learner(self):
        # self.learner = OFCLearner3DEndptPPF(1, dt=self.decoder.filt.dt, tau=self.tau)
        self.learn_flag = True

        kwargs = dict()
        dt = kwargs.pop('dt', 1./180)
        use_tau_unNat = self.tau
        self.tau = use_tau_unNat
        print "learner cost fn param: %g" % use_tau_unNat
        tau_scale = 28*use_tau_unNat/1000
        bin_num_ms = (dt/0.001)
        w_r = 3*tau_scale**2/2*(bin_num_ms)**2*26.61
        
        I = np.eye(3)
        zero_col = np.zeros([3, 1])
        zero_row = np.zeros([1, 3])
        zero = np.zeros([1,1])
        one = np.ones([1,1])
        A = np.bmat([[I, dt*I, zero_col], 
                     [0*I, 0*I, zero_col], 
                     [zero_row, zero_row, one]])
        B = np.bmat([[0*I], 
                     [dt/1e-3 * I],
                     [zero_row]])
        Q = np.mat(np.diag([1., 1, 1, 0, 0, 0, 0]))
        R = np.mat(np.diag([w_r, w_r, w_r]))
        
        F = feedback_controllers.LQRController.dlqr(A, B, Q, R)
        F_dict = dict(target=F, hold=F) 

        fb_ctrl = feedback_controllers.MultiModalLFC(A=A, B=B, F_dict=F_dict)

        batch_size = 1

        self.learner = clda.OFCLearner(batch_size, A, B, F_dict)
        # super(OFCLearner3DEndptPPF, self).__init__(batch_size, A, B, F_dict, *args, **kwargs)

        # Tell BMISystem that this learner wants the most recent output
        # of the decoder rather than the second most recent, to match MATLAB
        self.learner.input_state_index = 0






        #self.decoder.filt.W[3:6, 3:6] = np.eye(3) * 0.37182884 #0.4110

    # def _init_fb_controller(self):


    # def init(self):
    #     self._init_neural_encoder()
    #     self._init_fb_controller()
    #     # self.load_decoder()
    #     self.wait_time = 0
    #     self.pause = False
    #     super(SimCLDAControlMultiDispl2D_PPF, self).init()
    #     if self.updater is not None:
    #         self.updater.init(self.decoder)     


if args.alg == 'RML':
    if args.save:
        te = models.TaskEntry()
        sim_subj = models.Subject.objects.using('simulation').get(name='Simulation')
        te.subject = sim_subj
        te.task = models.Task.objects.using('simulation').get(name='clda_kf_cg_rml')
        te.sequence_id = 0
        te.save(using='simulation')
    gen = cursor_clda_tasks.SimCLDAControlMulti.sim_target_seq_generator_multi(8, 100)
    task = SimRML(gen)
elif args.alg == 'PPF':
    gen = cursor_clda_tasks.SimCLDAControlMulti.sim_target_seq_generator_multi(8, 100)
    task = SimCLDAControlMultiDispl2D_PPF(gen)
else:
    raise ValueError("Algorithm not recognized!")


self = task
task.init()
print 'task init called'
task.run()

if args.save:
    task.cleanup(dbq, te.id, subject=sim_subj, dbname='simulation')
