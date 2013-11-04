
from scipy.io import loadmat
import numpy as np
import math
import time
import cProfile
from riglib.bmi import train, clda, bmi
from tasks import bmimultitasks, generatorfunctions as genfns

reload(clda)
reload(train)
reload(bmi)


data_fname = '/Users/sgowda/Desktop/ppf_code_1023/jeev100713_VFB_PPF_B100_NS5_NU13_Z1_from1020_from1030_cont_rmv81_contData.mat'
data = loadmat(data_fname)

batch_idx = 0

spike_counts = data['spike_counts'].astype(np.float64)
intended_kin = data['intended_kin']
beta_hat = data['beta_hat']
aimPos = data['aimPos']
n_iter = data['n_iter'][0,0]
stimulant_index = data['stimulant_index']
param_noise_variances = data['param_noise_variances'].ravel()
stoch_beta_index = data['stoch_beta_index']
det_beta_index = data['det_beta_index']
cursor_kin = data['cursor_kin']

## make aim pos 3d
aimPos3D = np.vstack([aimPos[0,:], np.zeros(n_iter), aimPos[1,:]])

## Create the object representing the initial decoder
m_to_cm = 100.
cm_to_m = 0.01

init_beta = beta_hat[:,:,0]
init_beta = np.vstack([init_beta[1:,:], init_beta[0,:]]).T
units = np.vstack([(x,1) for x in range(init_beta.shape[0])])
fake_decoder = train._train_PPFDecoder_sim_known_beta(init_beta, units=units, dist_units='m')
decoder = train.load_PPFDecoder_from_mat_file(data_fname)
decoder.filt.C = fake_decoder.filt.C
#decoder.filt.W[3:6, 3:6] *= m_to_cm**2
decoder.n_subbins = 1
decoder.bmicount = 0

updater = clda.PPFContinuousBayesianUpdater(decoder, units='cm')

class TestPPFReconstruction(bmimultitasks.CLDAControlPPFContAdapt):
    def __init__(self, *args, **kwargs):
        super(TestPPFReconstruction, self).__init__(*args, **kwargs)
        self.idx = 0

    def load_decoder(self):
        self.decoder = decoder

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
        self.updater = clda.PPFContinuousBayesianUpdater(self.decoder, units='m')
        self.updater.dt = 0.005

    def get_spike_counts(self):
        sl = slice(3*self.idx, 3*(self.idx+1))
        self.idx += 1
        return spike_counts[:,sl]

    def get_cursor_location(self):
        self.target_location = aimPos3D[:,self.idx] #np.array([aimPos[0, idx], 0, aim
        if np.any(np.isnan(self.target_location)):
            self.state = 'wait'
        else:
            self.state = 'target'

        spike_counts = self.get_spike_counts()
        return self.call_decoder(spike_counts)

gen = genfns.sim_target_seq_generator_multi(8, 1000)
task = TestPPFReconstruction(gen)
task.init()

self = task
self.decoder['hand_px', 'hand_pz'] = cursor_kin[0:2,0]
#self.decoder['hand_px', 'hand_pz'] = 100*cursor_kin[0:2,0]
print self.decoder.get_state()
vel_error = np.zeros([2, n_iter])
pos_error = np.zeros([2, n_iter])
#n_iter = 3000
batch_idx = 0
for idx in range(1, n_iter):
    if idx % 1000 == 0: 
        print idx, np.max(np.abs(vel_error))
    #print 'true prev state', cursor_kin[:,idx-1]
    self.target_location = aimPos3D[:,idx]
    #self.target_location = m_to_cm*aimPos3D[:,idx]
    #print idx
    #print self.target_location
    if np.any(np.isnan(self.target_location)):
        self.state = 'wait'
    else:
        self.state = 'target'
        #print 'intended kin', intended_kin[:,batch_idx]
        batch_idx += 1

    spike_obs = spike_counts[:,idx].reshape(-1,1)
    # AdaptiveBMI call
    _, uf =  self.bmi_system(spike_obs, self.target_location,
        self.state, task_data=None, assist_level=self.current_assist_level,
        target_radius=self.target_radius, speed=-1,
        learn_flag=True, half_life=-1)
    
    vel_error[:,idx] = cursor_kin[2:4, idx] - self.decoder['hand_vx', 'hand_vz']#*cm_to_m
    pos_error[:,idx] = cursor_kin[0:2, idx] - self.decoder['hand_px', 'hand_pz']#*cm_to_m
    #print pos_error[:,idx]
    #print vel_error[:,idx]
    #print 
    beta_mat = beta_hat[:,:,idx]
    beta_dec = self.decoder.filt.C[:,[3,5,6]]
    beta_diff = np.array(beta_dec) - np.vstack([beta_mat[1:,:], beta_mat[0,:]]).T
    #print np.max(np.abs(beta_diff))

print np.max(np.abs(vel_error))
