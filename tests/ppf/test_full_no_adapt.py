
from scipy.io import loadmat
import numpy as np
import math
import time
import cProfile
from riglib.bmi import train, clda, bmi, ppfdecoder,extractor
from tasks import bmimultitasks, manualcontrolmultitasks, generatorfunctions as genfns
from features.simulation_features import SimHDF
from riglib.bmi.train import unit_conv
from tasks import cursor_clda_tasks

# reload(clda)
# reload(train)
# reload(bmi)
# reload(ppfdecoder)

class TestPPFReconstruction(bmimultitasks.BMIControlMulti):
    def __init__(self, *args, **kwargs):
        super(TestPPFReconstruction, self).__init__(*args, **kwargs)
        self.idx = 1
        self.n_subbins = 1
        self.task_data = SimHDF()
        self.hdf = SimHDF()
        self.learn_flag = True

        n_iter = kwargs.pop('n_iter')
        self.beta_error = np.zeros(n_iter)
        self.decoder_state = np.zeros([7, n_iter])
        self.neural_push = np.zeros([7, n_iter])
        self.P_est = np.zeros([7, 7, n_iter])
        self.decoder_error = np.zeros([7, n_iter])
        self.kwargs_init = kwargs
        self.state_units = kwargs.pop('state_units')
        self.spike_counts = kwargs.pop('spike_counts')

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
        decoder = train.load_PPFDecoder_from_mat_file(self.kwargs_init.pop('data_fname'))
        #decoder.filt.C = np.mat(init_beta) #fake_decoder.filt.C
        #decoder.n_subbins = 3
        #decoder.bmicount = 0
        self.decoder = decoder

        # Initialize the position of the decoder
        self.decoder.filt._init_state()
        self.decoder['hand_px', 'hand_pz'] = self.kwargs_init['cursor_kin'][0:2,0]*unit_conv('m', self.state_units)

    def create_learner(self):
        super(TestPPFReconstruction, self).create_learner()
        if self.kwargs_init['use_exact_F_int']:
            F_int_data = loadmat('/Users/sgowda/Desktop/ppf_code_1023/F_int.mat')
            F_int = F_int_data['F']
            alpha = F_int[0,0]
            beta = F_int[0,2]
            I = np.eye(3)
            F = np.mat(np.hstack([alpha*I, beta*I, np.zeros([3,1])]))
            self.learner.F_dict['target'] = F
            self.learner.F_dict['hold'] = F

    def get_spike_counts(self):
        return self.spike_counts[:,self.sl]

    def _update_target_loc(self):
        # Set the target location based on what was recorded in the .mat file
        self.target_location = self.kwargs_init['aimPos3D'][:,self.sl] * unit_conv('m', self.state_units)
        self.state = 'target'

    def get_cursor_location(self):
        if self.idx % 1000 == 0: 
            print(self.idx, np.max(np.abs(self.decoder_error[3:6,:])))

        self.current_assist_level = 0 # same indexing as MATLAB
        self.sl = slice(self.idx, self.idx+self.n_subbins)
        self._update_target_loc()
        spike_obs = self.get_spike_counts()
        
        self.call_decoder_output = self.call_decoder(spike_obs, np.zeros((7, 1)))
        self.decoder_state[:,self.sl] = self.call_decoder_output * unit_conv(self.state_units, 'm') 
        self.neural_push[:, self.sl] = self.decoder.filt.neural_push * unit_conv(self.state_units, 'm') 
        P = self.decoder.filt.P_est * unit_conv(self.state_units, 'm')**2
        self.P_est[:, :, self.sl] = P[:, :, np.newaxis]
        self.decoder_error[:,self.sl] = self.kwargs_init['cursor_kin_3d'][:,self.sl] - self.decoder_state[:,self.sl]
        self.beta_error[self.idx] = np.max(np.abs(self.decoder.filt.tomlab(unit_scale=100) - self.kwargs_init['beta_hat'][:,:,self.idx+self.n_subbins]))
        self.idx += self.n_subbins

    def create_feature_extractor(self):
        self.matsource = matfilesource(self.spike_counts)
        self.extractor = extractor.BinnedSpikeCountsExtractor(self.matsource)

    def _cycle(self):
        self.matsource.n += 1
        super(TestPPFReconstruction, self)._cycle()

class matfilesource(object):
    def __init__(self, spike_counts):
        self.spike_counts = spike_counts
        self.n = 0

    def get(self):
        print('soruc')
        return self.spike_counts[:, self.n]

def run_sim(data_fname=None, decoder_fname = None, n_iter2 = None, start_ix = 0):
    state_units = 'cm'
    #data_fname = '/Users/sgowda/Desktop/ppf_code_1023/assist_ex_data/jeev100413_VFB_PPF_B100_NS5_NU17_Z1_assist_ofc_contData.mat'
    #data_fname = '/Users/sgowda/Desktop/ppf_code_1023/jeev100713_VFB_PPF_B100_NS5_NU13_Z1_from1020_from1030_cont_rmv81_contData.mat'
    # if  data_fname is None:
    #     data_fname = '/home/lab/preeya/jeev_data_tmp/jeev080713_VFB_PPF_B100_NS5_NU18_Z1_assist_ofc_cont_cont_cont_swap50a15a_cont_swap58a50a113ab114ab_cont_swap124a125b_cont_cont_swap125ba_cont_cont_Barrier1fixData.mat'
    
    data = loadmat(data_fname)
    kwargs = dict()
    kwargs['spike_counts'] = data['spike_counts'][:, start_ix:].astype(np.float64)
    kwargs['intended_kin'] = data['intended_kin'][:, start_ix:]
    kwargs['beta_hat'] = data['beta_hat'][:, :, start_ix:]
    kwargs['aimPos'] = data['aimPos'][:, start_ix:]
    kwargs['n_iter'] = data['n_iter'][0,0] - start_ix
    kwargs['cursor_kin'] = data['cursor_kin'][:, start_ix:]
    kwargs['data_fname'] = data_fname
    kwargs['state_units'] = state_units

    ## convert to 3D kinematics
    kwargs['aimPos3D'] = np.vstack([kwargs['aimPos'][0,:], np.zeros(kwargs['n_iter']), kwargs['aimPos'][1,:]])
    cursor_kin_3d = np.zeros([7, kwargs['n_iter']])
    cursor_kin_3d[[0,2,3,5], :] = kwargs['cursor_kin']
    kwargs['cursor_kin_3d'] = cursor_kin_3d
    kwargs['use_exact_F_int'] = False
    kwargs['decoder_fname'] = decoder_fname

    #Run generator: 
    gen = manualcontrolmultitasks.ManualControlMulti.centerout_2D_discrete()
    task = TestPPFReconstruction(gen, **kwargs)
    task.init()

    self = task
    batch_idx = 0

    if n_iter2 is None or n_iter2 is 'max':
        n_iter = data['idx'][0, 0] - start_ix
    else:
        n_iter = n_iter2

    while self.idx < n_iter:
        st = time.time()
        self.get_cursor_location()
        #print time.time() - st
    print(np.max(np.abs(self.decoder_error[3:6,:])))

    return self, kwargs['cursor_kin']

