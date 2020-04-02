#!/usr/bin/python
"""
Simulation of CLDA control task
"""
## Imports

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


# reload(kfdecoder)
# reload(ppfdecoder)
reload(clda)
# reload(riglib.bmi)
# reload(riglib.bmi.train)
from tasks import tentaclebmitasks
reload(cursor_clda_tasks)
reload(tentaclebmitasks)

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

 

# from tasks.cursor_clda_tasks import OFCLearner3DEndptPPF
from riglib.experiment import traits
from riglib.bmi import feedback_controllers


class SimCLDAControlMultiDispl2D_PPF(SimTime, Autostart, WindowDispl2D, SimCosineTunedPointProc, SimPPFDecoderCursorShuffled, cursor_clda_tasks.CLDAControlPPFContAdapt):
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
        self.assist_level_time = 120.
        self.last_get_spike_counts_time = -1./60
        self.learn_flag = True

class SimCLDAControlMultiDispl2D_PPFRML(SimCLDAControlMultiDispl2D_PPF):
    half_life = (10., 120.)
    max_attempts = 1
    def create_updater(self):
        self.updater = clda.PPFRML()

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
elif args.alg == 'PPFRML':
    gen = cursor_clda_tasks.SimCLDAControlMulti.sim_target_seq_generator_multi(8, 100)
    task = SimCLDAControlMultiDispl2D_PPFRML(gen)
else:
    raise ValueError("Algorithm not recognized!")


self = task
task.init()
print('task init called')
task.run()

if args.save:
    task.cleanup(dbq, te.id, subject=sim_subj, dbname='simulation')
