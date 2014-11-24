
from scipy.io import loadmat
import numpy as np
import math
import time
import cProfile
from riglib.bmi import train, clda, bmi, ppfdecoder
from tasks import bmimultitasks, generatorfunctions as genfns
from features.simulation_features import SimHDF
from riglib.bmi.train import unit_conv

reload(clda)
reload(train)
reload(bmi)
reload(ppfdecoder)

# TODO clean up the way the decoder object is created from the MATLAB file

state_units = 'cm'

#data_fname = '/Users/sgowda/Desktop/ppf_code_1023/assist_ex_data/jeev100413_VFB_PPF_B100_NS5_NU17_Z1_assist_ofc_contData.mat'
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

use_exact_F_int = False

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
        '''
        Create the object for the initial decoder
        '''

        ## init_beta = beta_hat[:,:,0]
        ## init_beta = np.vstack([init_beta[1:,:], init_beta[0,:]]).T
        ## init_beta = ppfdecoder.PointProcessFilter.frommlab(beta_hat[:,:,0])
        ## init_beta[:,:-1] /= unit_conv('m', state_units)
        ## init_beta = train.inflate(init_beta, train.states_explaining_neural_activity_2D_vel_decoding, train.states_3D_endpt, axis=1)

        # units = np.vstack([(x,1) for x in range(init_beta.shape[0])])

        # TODO pull from beta_hat[:,:,0] if available
        ## fake_decoder = train._train_PPFDecoder_sim_known_beta(
        ##         init_beta, units=units, dist_units=state_units)

        decoder = train.load_PPFDecoder_from_mat_file(data_fname)
        #decoder.filt.C = np.mat(init_beta) #fake_decoder.filt.C
        #decoder.n_subbins = 3
        #decoder.bmicount = 0

        self.decoder = decoder

        # Initialize the position of the decoder
        self.decoder.filt._init_state()
        self.decoder['hand_px', 'hand_pz'] = cursor_kin[0:2,0]*unit_conv('m', state_units)

    def create_learner(self):
        super(TestPPFReconstruction, self).create_learner()
        if use_exact_F_int:
            F_int_data = loadmat('/Users/sgowda/Desktop/ppf_code_1023/F_int.mat')
            F_int = F_int_data['F']
            alpha = F_int[0,0]
            beta = F_int[0,2]
            I = np.eye(3)
            F = np.mat(np.hstack([alpha*I, beta*I, np.zeros([3,1])]))
            self.learner.F_dict['target'] = F
            self.learner.F_dict['hold'] = F

    def get_spike_counts(self):
        return spike_counts[:,self.sl]

    def _update_target_loc(self):
        # Set the target location based on what was recorded in the .mat file
        self.target_location = aimPos3D[:,self.sl] * unit_conv('m', state_units)
        self.state = 'target'

    def get_cursor_location(self):
        if self.idx % 1000 == 0: 
            print self.idx, np.max(np.abs(self.decoder_error[3:6,:]))

        self.current_assist_level = 0 # same indexing as MATLAB
        self.sl = slice(self.idx, self.idx+self.n_subbins)
        self._update_target_loc()
        spike_obs = self.get_spike_counts()
        
        self.call_decoder_output = self.call_decoder(spike_obs)
        self.decoder_state[:,self.sl] = self.call_decoder_output * unit_conv(state_units, 'm') 
        
        self.decoder_error[:,self.sl] = cursor_kin_3d[:,self.sl] - self.decoder_state[:,self.sl]
        self.beta_error[self.idx] = np.max(np.abs(self.decoder.filt.tomlab(unit_scale=100) - beta_hat[:,:,self.idx+self.n_subbins]))
        self.idx += self.n_subbins



gen = genfns.sim_target_seq_generator_multi(8, 1000)
task = TestPPFReconstruction(gen)
task.init()

self = task


batch_idx = 0
n_iter = 12000
while self.idx < n_iter:
    st = time.time()
    self.get_cursor_location()
    #print time.time() - st

print np.max(np.abs(self.decoder_error[3:6,:]))
