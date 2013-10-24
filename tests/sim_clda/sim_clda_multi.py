#!/usr/bin/python
"""
Simulation of CLDA control task
"""
## Imports
from __future__ import division
import os
import numpy as np
import multiprocessing as mp
from scipy.io import loadmat, savemat
from riglib.experiment.features import Autostart, SimHDF
import riglib.bmi
from riglib.bmi import kfdecoder, clda
from tasks import bmimultitasks, generatorfunctions as genfns

reload(kfdecoder)
reload(clda)
reload(riglib.bmi)
reload(riglib.bmi.train)

from riglib.stereo_opengl.window import WindowDispl2D

class SimCLDAControlMultiDispl2D(WindowDispl2D, bmimultitasks.SimCLDAControlMulti, Autostart):
    update_rate = 0.1
    def __init__(self, *args, **kwargs):
        self.target_radius = 1.8
        bmimultitasks.SimCLDAControlMulti.__init__(self, *args, **kwargs)
        self.batch_time = 0.1
        self.half_life  = 20.0

        self.origin_hold_time = 0.250
        self.terminus_hold_time = 0.250
        self.hdf = SimHDF()
        self.task_data = SimHDF()
        self.start_time = 0.
        self.loop_counter = 0
        self.assist_level = 0

    def create_updater(self):
        clda_input_queue = mp.Queue()
        clda_output_queue = mp.Queue()
        #self.updater = clda.KFRML(clda_input_queue, clda_output_queue, self.batch_time, self.half_life)
        self.updater = clda.KFOrthogonalPlantSmoothbatch(clda_input_queue, clda_output_queue, self.batch_time, self.half_life)

    def get_time(self):
        return self.loop_counter * 1./60

    def loop_step(self):
        self.loop_counter += 1

class SimBMIControlMultiDispl2D_PPF(WindowDispl2D, bmimultitasks.SimBMIControlMulti, Autostart):
    '''
    Sim of PPF where beta used to generate the spikes is known to the decoder
    '''
    def __init__(self, *args, **kwargs):
        self.target_radius = 1.8
        bmimultitasks.SimBMIControlMulti.__init__(self, *args, **kwargs)
        self.batch_time = 5
        self.half_life  = 20.0

        self.origin_hold_time = 0.250
        self.terminus_hold_time = 0.250
        self.hdf = SimHDF()
        self.task_data = SimHDF()
        self.start_time = 0.
        self.loop_counter = 0
        self.assist_level = 0

    def load_decoder(self):
        N = 10000
        fname = '/Users/sgowda/code/bmi3d/tests/ppf/sample_spikes_and_kinematics_%d.mat' % N 
        data = loadmat(fname)

        dt = 1./180
        states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
        decoding_states = ['hand_vx', 'hand_vz', 'offset'] 

        beta = data['beta']
        beta = np.vstack([beta[1:, :], beta[0,:]]).T
        beta_dec = riglib.bmi.train.inflate(beta, decoding_states, states, axis=1)

        self.decoder = riglib.bmi.train._train_PPFDecoder_sim_known_beta(beta_dec, 
                self.encoder.units, dt=dt, dist_units='cm')
        
    def init(self):
        '''
        Instantiate simulation decoder
        '''
        bmimultitasks.SimBMIControlMulti.init(self)
        self.load_decoder()

        self.n_subbins = 3
        self.last_get_spike_counts_time = 0#-1./60

    def _init_neural_encoder(self):
        from riglib.bmi import sim_neurons
        sim_encoder_fname = os.path.join(os.getenv('HOME'), 'code/bmi3d/tests/ppf', 'sample_spikes_and_kinematics_10000.mat')
        self.encoder = sim_neurons.load_ppf_encoder_2D_vel_tuning_clda_sim(sim_encoder_fname, dt=1./60) #CosEnc(fname=sim_encoder_fname, return_ts=True)

    def get_cursor_location(self):
        spike_counts = self.get_spike_counts()
        self.call_decoder(spike_counts)
        return self.decoder['hand_px', 'hand_py', 'hand_pz']

    def get_time(self):
        return self.loop_counter * 1./60

    def loop_step(self):
        self.loop_counter += 1


gen = genfns.sim_target_seq_generator_multi(8, 1000)
task = SimBMIControlMultiDispl2D_PPF(gen)
#task = SimCLDAControlMultiDispl2D(gen)
task.init()
task.run()
