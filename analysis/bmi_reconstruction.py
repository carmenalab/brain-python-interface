'''Utilities to reconstruct BMI trajectories'''
import numpy as np
import os
import tables
import pickle

#import bmimultitasks, cursor_clda_tasks
from riglib.bmi.bmi import BMILoop
from riglib.bmi import extractor, clda, feedback_controllers, goal_calculators
from riglib.experiment import Experiment, traits
from features.simulation_features import SimHDF
from riglib import plants

def sim_target_seq_generator_multi(n_targs, n_trials):
    '''
    Simulated generator for simulations of the BMIControlMulti and CLDAControlMulti tasks
    '''
    center = np.zeros(2)
    pi = np.pi
    targets = 8*np.vstack([[np.cos(pi/4*k), np.sin(pi/4*k)] for k in range(8)])

    target_inds = np.random.randint(0, n_targs, n_trials)
    target_inds[0:n_targs] = np.arange(min(n_targs, n_trials))
    for k in range(n_trials):
        targ = targets[target_inds[k], :]
        yield np.array([[center[0], 0, center[1]],
                        [targ[0], 0, targ[1]]])


class BMIReconstruction(BMILoop, Experiment):
    fps = 60
    def __init__(self, n_iter=None, entry_id=None, hdf_filename=None, decoder_filename=None, params=dict(), *args, **kwargs):
        if entry_id is None and (hdf_filename is None or decoder_filename is None):
            raise ValueError("Not enough data to reconstruct a BMI! Specify a database entry OR an HDF file + decoder file")
        if entry_id is not None:
            from db import dbfunctions as dbfn
            te = dbfn.TaskEntry(entry_id)
            self.hdf_ref = te.hdf
            self.decoder = te.decoder
            self.params = te.params

            if self.hdf_ref is None:
                raise ValueError("Database is unable to locate HDF file!")
            if self.decoder is None:
                raise ValueError("Database is unable to locate HDF file!")
        elif hdf_filename is not None:
            self.hdf_ref = tables.open_file(hdf_filename)
            self.decoder = pickle.load(open(decoder_filename, 'rb'), encoding='latin1') # extra args to get py3 to read py2 pickles
            self.params = params


        self.n_iter = min(n_iter, len(self.hdf_ref.root.task))

        try:
            self.starting_pos = self.hdf_ref.root.task[0]['decoder_state'][0:3,0]
        except:
            # The statement above appears to not always work...
            self.starting_pos = self.hdf_ref.root.task[0]['cursor'] # #(0, 0, 0)

        # if 'plant_type' in te.params:
        #     self.plant_type = te.params['plant_type']
        # elif 'arm_class' in te.params:
        #     plant_type = te.params['arm_class']
        #     if plant_type == 'CursorPlant':
        #         self.plant_type = 'cursor_14x14'
        #     else:
        #         self.plant_type = plant_type
        # else:
        #     self.plant_type = 'cursor_14x14'

        # TODO overly specific
        self.plant = plants.CursorPlant(endpt_bounds=(-14, 14, 0., 0., -14, 14))

        ## Set the target radius because the old assist method changes the assist speed
        # when the cursor is inside the target
        self.target_radius = params['target_radius']
        self.cursor_radius = params['cursor_radius']
        self.assist_level = params['assist_level']

        self.idx = 0
        gen = sim_target_seq_generator_multi(8, 1000)

        super(BMIReconstruction, self).__init__(gen, *args, **kwargs)

        self.hdf = SimHDF()
        self.learn_flag = True

        task_msgs = self.hdf_ref.root.task_msgs[:]
        self.update_bmi_msgs = task_msgs[task_msgs['msg'] == 'update_bmi']
        task_msgs = list(filter(lambda x: x['msg'] not in ['update_bmi'], task_msgs))
        # print task_msgs
        self.task_state = np.array([None]*n_iter)
        for msg, next_msg in zip(task_msgs[:-1], task_msgs[1:]):
            self.task_state[msg['time']:next_msg['time']] = msg['msg']

        self.update_bmi_inds = np.zeros(len(self.hdf_ref.root.task))
        self.update_bmi_inds[self.update_bmi_msgs['time']] = 1
        self.recon_update_bmi_inds = np.zeros(len(self.hdf_ref.root.task))

        self.target_hold_msgs = list(filter(lambda x: x['msg'] in ['target', 'hold'], self.hdf_ref.root.task_msgs[:]))

    def calc_recon_error(self, n_iter_betw_fb=100000, **kwargs):
        """Main function to call for reconstruction"""
        saved_state = self.hdf_ref.root.task[:]['decoder_state']
        while self.idx < self.n_iter:
            # print self.current_assist_level
            self.get_cursor_location(**kwargs)

            if self.idx % n_iter_betw_fb == 0:
                if saved_state.dtype == np.float32:
                    error = saved_state[:self.idx,:,-1] - np.float32(self.decoder_state[:self.idx,:,-1])
                else:
                    error = saved_state[:self.idx,:,-1] - self.decoder_state[:self.idx,:,-1]
                print("Error after %d iterations" % self.idx, np.max(np.abs(error)))

        if saved_state.dtype == np.float32:
            error = saved_state[:self.n_iter,:,-1] - np.float32(self.decoder_state[:self.n_iter,:,-1])
        else:
            error = saved_state[:self.n_iter,:,-1] - self.decoder_state[:self.n_iter,:,-1]

        return error

    def init_decoder_state(self):
        '''
        Initialize the state of the decoder to match the initial state of the plant
        '''
        init_decoder_state = self.hdf_ref.root.task[0]['decoder_state']
        if init_decoder_state.shape[1] > 1:
            init_decoder_state = init_decoder_state[:,0].reshape(-1,1)

        self.init_decoder_mean = init_decoder_state
        self.decoder.filt.state.mean = self.init_decoder_mean

        self.decoder.set_call_rate(self.fps)

    def create_feature_extractor(self):
        '''
        Create the feature extractor object
        '''
        self.extractor = extractor.ReplaySpikeCountsExtractor(self.hdf_ref.root.task,
            source='spike_counts', units=self.decoder.units, cycle_rate=self.fps)
        self._add_feature_extractor_dtype()

    def load_decoder(self):
        '''
        Create the object for the initial decoder
        '''
        self.n_subbins = self.decoder.n_subbins
        self.decoder_state = np.zeros([self.n_iter, self.decoder.n_states, self.n_subbins])

    def get_spike_counts(self):
        return self.hdf_ref.root.task[self.idx]['spike_counts']

    def _update_target_loc(self):
        self.target_location = self.hdf_ref.root.task[self.idx]['target']
        self.state = self.task_state[self.idx]

    def get_cursor_location(self, verbose=False):
        if self.idx % 1000 == 0 and verbose: print(self.idx)

        self.current_assist_level = self.hdf_ref.root.task[self.idx]['assist_level'][0]
        try:
            self.current_half_life = self.hdf_ref.root.task[self.idx]['half_life'][0]
        except:
            self.current_half_life = 0
        self._update_target_loc()

        self.call_decoder_output = self.move_plant(half_life=self.current_half_life)
        if verbose:
            print(self.call_decoder_output)
            print(self.hdf_ref.root.task[self.idx]['decoder_state'])
            print()
        self.decoder_state[self.idx] = self.call_decoder_output
        self.idx += 1

    def call_decoder(self, neural_obs, target_state, **kwargs):
        '''
        Run the decoder computations

        Parameters
        ----------
        neural_obs : np.array of shape (n_features, n_subbins)
            n_features is the number of neural features the decoder is expecting to decode from.
            n_subbins is the number of simultaneous observations which will be decoded (typically 1)
        target_state: np.array of shape (n_states, 1)
            The current optimal state to be in to accomplish the task. In this function call, this gets
            used when adapting the decoder using CLDA
        '''
        # Get the decoder output
        decoder_output, update_flag = self.bmi_system(neural_obs, target_state, self.state, learn_flag=self.learn_flag, **kwargs)
        if update_flag:
            # send msg to hdf file to indicate decoder update
            self.hdf_ref.sendMsg("update_bmi")
            self.recon_update_bmi_inds[self.idx] = 1

        return decoder_output

    def get_time(self):
        t = self.idx * 1./self.fps
        return t


