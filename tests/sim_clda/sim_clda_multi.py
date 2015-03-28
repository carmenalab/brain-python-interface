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
from features.simulation_features import SimHDF, SimTime
from features.hdf_features import SaveHDF

import riglib.bmi
from riglib.bmi import train, kfdecoder, clda, ppfdecoder
from tasks import bmimultitasks, generatorfunctions as genfns
from riglib.stereo_opengl.window import WindowDispl2D
from tasks import cursor_clda_tasks

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


class PointProcNeuralSim(object):
    def _init_neural_encoder(self):
        from riglib.bmi import sim_neurons
        sim_encoder_fname = os.path.join(os.getenv('HOME'), 'code/bmi3d/tests/ppf', 'sample_spikes_and_kinematics_10000.mat')
        self.encoder = sim_neurons.load_ppf_encoder_2D_vel_tuning_clda_sim(sim_encoder_fname, dt=1./60) #CosEnc(fname=sim_encoder_fname, return_ts=True)

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

class SimCLDAControlMultiDispl2D(SaveHDF, Autostart, SimTime, WindowDispl2D, cursor_clda_tasks.SimCLDAControlMulti):
    update_rate = 0.1
    starting_pos = (0., 0., 0.)
    rand_start = (0., 0.)
    def __init__(self, *args, **kwargs):
        self.target_radius = 1.8
        super(SimCLDAControlMultiDispl2D, self).__init__(*args, **kwargs)
        self.batch_time = 10.
        self.half_life  = 20.0, 20.0

    def create_updater(self):
        # clda_input_queue = mp.Queue()
        # clda_output_queue = mp.Queue()
        self.updater = clda.KFOrthogonalPlantSmoothbatch(self.batch_time, self.half_life[0])
        
class SimRML(SimCLDAControlMultiDispl2D):
    assist_level = (0., 0.)
    plant_type = 'cursor_14x14'
    def __init__(self, *args, **kwargs):
        super(SimRML, self).__init__(*args, **kwargs)
        self.batch_time = 0.1
        self.half_life  = (20.0, 20.0)
        self.starting_pos = (0., 0., 0.)
        self.assist_time = 15.

    def create_updater(self):
        self.updater = clda.KFRML(self.batch_time, self.half_life[0])

    def load_decoder(self):
        from db import namelist
        ssm = namelist.endpt_2D_state_space
        self.decoder = train._train_KFDecoder_2D_sim(ssm, self.encoder.get_units())

    def _cycle(self):
        super(SimRML, self)._cycle()


 
class SimCLDAControlMultiDispl2D_PPF(cursor_clda_tasks.CLDAControlPPFContAdapt, SimCLDAControlMultiDispl2D):
    def __init__(self, *args, **kwargs):
        super(SimCLDAControlMultiDispl2D_PPF, self).__init__(*args, **kwargs)
        self.batch_time = 1./10 #60.  # TODO 10 Hz running seems to be hardcoded somewhere
        self.assist_level = 1., 0.
        self.assist_time = 60.
        self.last_get_spike_counts_time = -1./60
        self.learn_flag = True

    def load_decoder(self):
        decoder_fname = '/storage/decoders/grom20150102_14_BPPF01021655.pkl'
        decoder = pickle.load(open(decoder_fname))

        beta = decoder.filt.C
        self.init_beta = beta.copy()

        inds = np.arange(beta.shape[0])
        np.random.shuffle(inds)
        beta_dec = beta[inds, :]

        self.decoder = riglib.bmi.train._train_PPFDecoder_sim_known_beta(beta_dec, 
                self.encoder.units, dt=dt, dist_units='cm')
        #self.decoder.filt.W[3:6, 3:6] = np.eye(3) * 0.37182884 #0.4110

    def _init_neural_encoder(self):
        from riglib.bmi import sim_neurons
        decoder_fname = '/storage/decoders/grom20150102_14_BPPF01021655.pkl'
        decoder = pickle.load(open(decoder_fname))

        beta = decoder.filt.C
        dt = decoder.filt.dt 
        self.encoder = sim_neurons.CLDASimPointProcessEnsemble(beta[:,[3,5,6]], dt)        
        # sim_encoder_fname = os.path.join(os.getenv('HOME'), 'code/bmi3d/tests/ppf', 'sample_spikes_and_kinematics_10000.mat')
        # self.encoder = sim_neurons.load_ppf_encoder_2D_vel_tuning_clda_sim(sim_encoder_fname, dt=1./60) #CosEnc(fname=sim_encoder_fname, return_ts=True)
    
    def get_time(self):
        return SimCLDAControlMultiDispl2D.get_time(self)



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
    gen = cursor_clda_tasks.SimCLDAControlMulti.sim_target_seq_generator_multi(8, 1)
    task = SimCLDAControlMultiDispl2D_PPF(gen)
else:
    raise ValueError("Algorithm not recognized!")


self = task
task.init()
print 'task init called'
task.run()

if args.save:
    task.cleanup(dbq, te.id, subject=sim_subj, dbname='simulation')
