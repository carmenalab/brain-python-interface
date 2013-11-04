
from scipy.io import loadmat
import numpy as np
import math
import time
import cProfile
from riglib.bmi import train, clda, bmi, ppfdecoder
from tasks import bmimultitasks, generatorfunctions as genfns
from riglib.experiment.features import SimHDF

reload(clda)
reload(train)
reload(bmi)
reload(ppfdecoder)

# TODO clean up the way the decoder object is created from the MATLAB file

state_units = 'cm'
def conv(starting_unit, ending_unit):
    if starting_unit == ending_unit:
        return 1
    elif (starting_unit, ending_unit) == ('cm', 'm'):
        return 0.01
    elif (starting_unit, ending_unit) == ('m', 'cm'):
        return 100


data_fname = '/Users/sgowda/Desktop/ppf_code_1023/jeev100713_VFB_PPF_B100_NS5_NU13_Z1_from1020_from1030_cont_rmv81_contData.mat'
data = loadmat(data_fname)

spike_counts = data['spike_counts'].astype(np.float64)
intended_kin = data['intended_kin']
beta_hat = data['beta_hat']
aimPos = data['aimPos']
n_iter = data['n_iter'][0,0]
cursor_kin = data['cursor_kin']

## convert to 3D kinematics
aimPos3D = np.vstack([aimPos[0,:], np.zeros(n_iter), aimPos[1,:]])
cursor_kin_3d = np.zeros([7, n_iter])
cursor_kin_3d[[0,2,3,5], :] = cursor_kin

class TestPPFReconstruction(bmimultitasks.CLDAControlPPFContAdapt):
    def __init__(self, *args, **kwargs):
        super(TestPPFReconstruction, self).__init__(*args, **kwargs)
        self.idx = 1
        self.task_data = SimHDF()
        self.hdf = SimHDF()
        self.learn_flag = True

        self.beta_error = np.zeros(n_iter)
        self.decoder_state = np.zeros([7, n_iter])
        self.decoder_error = np.zeros([7, n_iter])

    def load_decoder(self):
        ## Create the object representing the initial decoder
        init_beta = beta_hat[:,:,0]
        init_beta = np.vstack([init_beta[1:,:], init_beta[0,:]]).T
        units = np.vstack([(x,1) for x in range(init_beta.shape[0])])
        fake_decoder = train._train_PPFDecoder_sim_known_beta(
                init_beta, units=units, dist_units=state_units)
        decoder = train.load_PPFDecoder_from_mat_file(data_fname)
        decoder.filt.C = fake_decoder.filt.C
        if state_units == 'cm':
            decoder.filt.W[3:6, 3:6] *= conv('m', state_units)**2
        decoder.n_subbins = 1
        decoder.bmicount = 0

        self.decoder = decoder
        self.decoder.filt._init_state()

    def create_learner(self):
        self.learner = clda.OFCLearner3DEndptPPF(
                            self.decoder.n_subbins, dt=0.005)
        F_int_data = loadmat('/Users/sgowda/Desktop/ppf_code_1023/F_int.mat')
        F_int = F_int_data['F']
        alpha = F_int[0,0]
        beta = F_int[0,2]
        I = np.eye(3)
        F = np.mat(np.hstack([alpha*I, beta*I, np.zeros([3,1])]))
        self.learner.F_dict['target'] = F
        self.learner.F_dict['hold'] = F

    def create_updater(self):
        self.batch_size = self.decoder.n_subbins
        self.create_learner() # Recreate learner
        self.updater = clda.PPFContinuousBayesianUpdater(self.decoder, units=state_units)
        self.updater.dt = 0.005

    def get_spike_counts(self):
        return spike_counts[:,self.sl]

    def _update_target_loc(self):
        # Set the target location based on what was recorded in the .mat file
        self.target_location = aimPos3D[:,self.sl] * conv('m', state_units)
        #self.target_location.reshape(-1,1)
        if np.any(np.isnan(self.target_location)):
            self.state = 'wait'
        else:
            self.state = 'target'

    def _update_sl(self):
        # Determine which time data to slice
        self.sl = slice(self.idx, self.idx+self.n_subbins)

    def get_cursor_location(self):
        if self.idx % 1000 == 0: 
            print self.idx, np.max(np.abs(self.decoder_error[3:6,:]))

        ## # Set the target location based on what was recorded in the .mat file
        ## self.target_location = aimPos3D[:,self.idx] * conv('m', state_units)
        ## if np.any(np.isnan(self.target_location)):
        ##     self.state = 'wait'
        ## else:
        ##     self.state = 'target'
        ##     batch_idx += 1

        self._update_sl()
        self._update_target_loc()
        spike_obs = self.get_spike_counts()

        #spike_obs = spike_counts[:,self.idx].reshape(-1,1)

        # AdaptiveBMI call
        ## _, uf =  self.bmi_system(spike_obs, self.target_location,
        ##     self.state, task_data=None, assist_level=self.current_assist_level,
        ##     target_radius=self.target_radius, speed=-1,
        ##     learn_flag=True, half_life=-1)
        self.call_decoder(spike_obs)

        self.decoder_state[:,self.idx] = self.decoder.get_state()*conv(state_units, 'm')
        self.decoder_error[:,self.idx] = cursor_kin_3d[:,self.idx] - self.decoder_state[:,self.idx]
        self.beta_error[self.idx] = np.max(np.abs(self.decoder.filt.tomlab(unit_scale=100) - beta_hat[:,:,self.idx+self.n_subbins]))
        self.idx += self.n_subbins



gen = genfns.sim_target_seq_generator_multi(8, 1000)
task = TestPPFReconstruction(gen)
task.init()

self = task

# Initialize the position of the decoder
self.decoder['hand_px', 'hand_pz'] = cursor_kin[0:2,0]*conv('m', state_units)

batch_idx = 0
while self.idx < n_iter:
#for idx in range(1, n_iter):
    #if self.idx % 1000 == 0: 
    #    print self.idx, np.max(np.abs(self.decoder_error[3:6,:]))

    self.get_cursor_location()
    #### ## # Set the target location based on what was recorded in the .mat file
    #### ## self.target_location = aimPos3D[:,self.idx] * conv('m', state_units)
    #### ## if np.any(np.isnan(self.target_location)):
    #### ##     self.state = 'wait'
    #### ## else:
    #### ##     self.state = 'target'
    #### ##     batch_idx += 1

    #### self._update_target_loc()
    #### self._update_sl()
    #### spike_obs = self.get_spike_counts()

    #### #spike_obs = spike_counts[:,self.idx].reshape(-1,1)

    #### # AdaptiveBMI call
    #### ## _, uf =  self.bmi_system(spike_obs, self.target_location,
    #### ##     self.state, task_data=None, assist_level=self.current_assist_level,
    #### ##     target_radius=self.target_radius, speed=-1,
    #### ##     learn_flag=True, half_life=-1)
    #### self.call_decoder(spike_obs)

    #### self.decoder_state[:,self.idx] = self.decoder.get_state()*conv(state_units, 'm')
    #### self.decoder_error[:,self.idx] = cursor_kin_3d[:,self.idx] - self.decoder_state[:,self.idx]
    #### self.beta_error[self.idx] = np.max(np.abs(self.decoder.filt.tomlab(unit_scale=100) - beta_hat[:,:,self.idx]))
    #### self.idx += 1

print np.max(np.abs(self.decoder_error[3:6,:]))
