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
from riglib.bmi import kfdecoder, clda, ppfdecoder
from tasks import bmimultitasks, generatorfunctions as genfns

reload(kfdecoder)
reload(ppfdecoder)
reload(clda)
reload(riglib.bmi)
reload(riglib.bmi.train)

from riglib.stereo_opengl.window import WindowDispl2D

class SimTime(object):
    def __init__(self):
        self.start_time = 0.
        self.loop_counter = 0

    def get_time(self):
        try:
            return self.loop_counter * 1./60
        except:
            # loop_counter has not been initialized yet, return 0
            return 0

    def loop_step(self):
        self.loop_counter += 1

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
        print batch_size
        half_life = 120.
        rho = np.exp(np.log(0.5) / (half_life/batch_time))
        print rho
        
        #self.updater = clda.PPFContinuousBayesianUpdater(self.decoder)
        self.updater = clda.PPFSmoothbatchSingleThread()
        self.updater.rho = rho
        self.updater.batch_time = batch_time

class PointProcContinuous(object):
    def create_updater(self):
        self.updater = clda.PPFContinuousBayesianUpdater(self.decoder)

class SimCLDAControlMultiDispl2D(SimTime, WindowDispl2D, bmimultitasks.SimCLDAControlMulti, PointProcNeuralSim, Autostart):
    update_rate = 0.1
    def __init__(self, *args, **kwargs):
        self.target_radius = 1.8
        bmimultitasks.SimCLDAControlMulti.__init__(self, *args, **kwargs)
        self.batch_time = 10.
        self.half_life  = 20.0, 20.0

        self.hdf = SimHDF()
        self.task_data = SimHDF()
        SimTime.__init__(self)

    def create_updater(self):
        clda_input_queue = mp.Queue()
        clda_output_queue = mp.Queue()
        #self.updater = clda.KFRML(clda_input_queue, clda_output_queue, self.batch_time, self.half_life)
        self.updater = clda.KFOrthogonalPlantSmoothbatch(clda_input_queue, clda_output_queue, self.batch_time, self.half_life)

    ## def create_learner(self):
    ##     dt = 0.1
    ##     A = np.mat([[1., 0, 0, dt, 0, 0, 0], 
    ##                 [0., 0, 0, 0,  0, 0, 0],
    ##                 [0., 0, 1, 0, 0, dt, 0],
    ##                 [0., 0, 0, 0, 0,  0, 0],
    ##                 [0., 0, 0, 0, 0,  0, 0],
    ##                 [0., 0, 0, 0, 0,  0, 0],
    ##                 [0., 0, 0, 0, 0,  0, 1]])

    ##     I = np.mat(np.eye(3))
    ##     B = np.vstack([0*I, I, np.zeros([1,3])])
    ##     F_target = np.hstack([I, 0*I, np.zeros([3,1])])
    ##     F_hold = np.hstack([0*I, 0*I, np.zeros([3,1])])
    ##     F_dict = dict(hold=F_hold, target=F_target)
    ##     self.learner = clda.OFCLearner(self.batch_size, A, B, F_dict)
    ##     self.learn_flag = True
        

#SimTime, WindowDispl2D, bmimultitasks.SimCLDAControlMulti, PointProcNeuralSim, Autostart

class SimCLDAControlMultiDispl2D_PPF(bmimultitasks.CLDAControlPPFContAdapt, SimCLDAControlMultiDispl2D):
    def __init__(self, *args, **kwargs):
        super(SimCLDAControlMultiDispl2D_PPF, self).__init__(*args, **kwargs)
        self.batch_time = 1./10 #60.  # TODO 10 Hz running seems to be hardcoded somewhere
        self.assist_level = 1., 0.
        self.assist_time = 60.
        self.last_get_spike_counts_time = -1./60
        self.learn_flag = True

    def load_decoder(self):
        N = 10000
        fname = '/Users/sgowda/code/bmi3d/tests/ppf/sample_spikes_and_kinematics_%d.mat' % N 
        data = loadmat(fname)

        dt = 1./180
        #states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
        #decoding_states = ['hand_vx', 'hand_vz', 'offset'] 

        beta = data['beta']
        beta = np.vstack([beta[1:, :], beta[0,:]]).T
        #beta_dec = riglib.bmi.train.inflate(beta, decoding_states, states, axis=1)
        #beta_dec[:,[3,5]] *= 1

        self.init_beta = beta.copy()

        inds = np.arange(beta.shape[0])
        np.random.shuffle(inds)
        beta_dec = beta[inds, :]

        self.decoder = riglib.bmi.train._train_PPFDecoder_sim_known_beta(beta_dec, 
                self.encoder.units, dt=dt, dist_units='cm')
        #self.decoder.filt.W[3:6, 3:6] = np.eye(3) * 0.37182884 #0.4110

    def _init_neural_encoder(self):
        from riglib.bmi import sim_neurons
        sim_encoder_fname = os.path.join(os.getenv('HOME'), 'code/bmi3d/tests/ppf', 'sample_spikes_and_kinematics_10000.mat')
        self.encoder = sim_neurons.load_ppf_encoder_2D_vel_tuning_clda_sim(sim_encoder_fname, dt=1./60) #CosEnc(fname=sim_encoder_fname, return_ts=True)
    
    def get_time(self):
        return SimCLDAControlMultiDispl2D.get_time(self)


class SimRML(SimCLDAControlMultiDispl2D):
    def __init__(self, *args, **kwargs):
        super(SimRML, self).__init__(*args, **kwargs)
        self.batch_time = 0.1
        self.half_life  = (20.0, 20.0)

    def create_updater(self):
        self.updater = clda.KFRML(None, None, self.batch_time, self.half_life[0])

gen = genfns.sim_target_seq_generator_multi(8, 1000)
task = SimRML(gen)
#task = SimCLDAControlMultiDispl2D_PPF(gen)
#task = SimCLDAControlMultiDispl2D(gen)
task.init()
task.run()
