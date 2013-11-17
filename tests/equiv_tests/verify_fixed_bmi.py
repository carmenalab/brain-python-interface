#!/usr/bin/python
'''
Test case to check that the current state of the code is able to reconstruct a TaskEntry using the BMIControlMulti task '''
from db import dbfunctions as dbfn
from tasks import performance
from scipy.io import loadmat
import numpy as np
import math
import time
import cProfile
from riglib.bmi import train, clda, bmi, ppfdecoder
from tasks import bmimultitasks, generatorfunctions as genfns
from riglib.experiment.features import SimHDF
from riglib.bmi.train import unit_conv
from itertools import izip

reload(clda)
reload(train)
reload(bmi)
reload(ppfdecoder)

idx = 2295
te = performance._get_te(idx)
n_iter = len(te.hdf.root.task)

class BMIReconstruction(bmimultitasks.BMIControlMulti):
    def __init__(self, *args, **kwargs):
        super(BMIReconstruction, self).__init__(*args, **kwargs)
        self.idx = 0
        self.task_data = SimHDF()
        self.hdf = SimHDF()
        self.learn_flag = True

        task_msgs = te.hdf.root.task_msgs[:]
        # TODO filter out 'update_bmi' msgs
        self.task_state = np.array([None]*n_iter)
        for msg, next_msg in izip(task_msgs[:-1], task_msgs[1:]):
            self.task_state[msg['time']:next_msg['time']] = msg['msg']

    def load_decoder(self):
        '''
        Create the object for the initial decoder
        '''
        self.decoder = te.decoder
        self.n_subbins = self.decoder.n_subbins
        self.decoder_state = np.zeros([n_iter, 7, self.n_subbins])

    def get_spike_counts(self):
        return te.hdf.root.task[self.idx]['spike_counts']

    def _update_target_loc(self):
        #self.target_location = te.hdf.root.task[self.idx]['target']
        self.target_location = None
        self.state = self.task_state[self.idx] #te.hdf.root.task_msgs[:]

    def get_cursor_location(self):
        if self.idx % 1000 == 0: 
            print self.idx

        self.current_assist_level = 0 # same indexing as MATLAB
        self._update_target_loc()
        spike_obs = self.get_spike_counts()
        
        self.call_decoder_output = self.call_decoder(spike_obs.astype(np.float64))
        self.decoder_state[self.idx] = self.call_decoder_output
        self.idx += 1


gen = genfns.sim_target_seq_generator_multi(8, 1000)

task = BMIReconstruction(gen)
task.init()

self = task

batch_idx = 0
while self.idx < n_iter:
    st = time.time()
    self.get_cursor_location()
    #print time.time() - st

cursor = te.hdf.root.task[:]['cursor']
if cursor.dtype == np.float32:
    error = cursor[:n_iter] - np.float32(self.decoder_state[:n_iter,0:3,-1])
else:
    error = cursor[:n_iter] - self.decoder_state[:n_iter,0:3,-1]
print "Recon error", np.max(np.abs(error))
