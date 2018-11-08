#!/usr/bin/python
'''
Test case to check that the current state of the code is able to reconstruct a TaskEntry using the BMIControlMulti task '''
from db import dbfunctions as dbfn
from analysis import performance
from scipy.io import loadmat
import numpy as np
import math
import time
import cProfile
from riglib.bmi import train, clda, bmi, ppfdecoder, extractor
from tasks import bmimultitasks, generatorfunctions as genfns
from riglib.bmi.train import unit_conv

from tasks import bmi_recon_tasks

reload(clda)
reload(train)
reload(bmi)
reload(ppfdecoder)


idx = 2306
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-b", "--block", dest="block", help="Database ID number of block to run on", type=int, metavar="FILE", default=2306)

(options, args) = parser.parse_args()
idx = options.block
te = performance._get_te(idx)
print(te)
T = len(te.hdf.root.task)
n_iter = 100 #T
#n_iter = 15782

task_msgs = te.hdf.root.task_msgs[:]
update_bmi_msgs = task_msgs[task_msgs['msg'] == 'update_bmi']
state_transitions = task_msgs[~(task_msgs['msg'] == 'update_bmi')]

# class CLDAPPFReconstruction(bmi_recon_tasks.BMIReconstruction):
#     pass

# class CLDAPPFReconstruction(bmimultitasks.CLDAControlPPFContAdapt):
#     def __init__(self, *args, **kwargs):
#         super(CLDAPPFReconstruction, self).__init__(*args, **kwargs)
#         self.idx = 0
#         self.task_data = SimHDF()
#         self.hdf = SimHDF()
#         self.learn_flag = True

#         task_msgs = te.hdf.root.task_msgs[:]
#         # TODO filter out 'update_bmi' msgs
#         task_msgs = state_transitions #task_msgs[~(task_msgs['msg'] == 'update_bmi')]
#         self.task_state = np.array([None]*T)
#         for msg, next_msg in izip(task_msgs[:-1], task_msgs[1:]):
#             if msg['time'] == next_msg['time']:
#                 print msg, next_msg
#                 next_msg['time'] += 1
#                 print msg, next_msg
#             if msg['msg'] == 'targ_transition' and next_msg['time'] - msg['time'] > 1:
#                 print msg, next_msg
#                 next_msg['time'] = msg['time'] + 1
#                 print msg, next_msg
#             self.task_state[msg['time']:next_msg['time']] = msg['msg']

#         self.tau = te.params['tau']

#     def load_decoder(self):
#         '''
#         Create the object for the initial decoder
#         '''
#         self.decoder = te.decoder
#         self.n_subbins = self.decoder.n_subbins
#         self.decoder_state = np.zeros([T, 7, self.n_subbins])

#     def get_spike_counts(self):
#         return te.hdf.root.task[self.idx]['spike_counts']

#     def _update_target_loc(self):
#         self.target_location = te.hdf.root.task[self.idx]['target']
#         #self.target_location = None
#         if self.idx in update_bmi_msgs['time']:
#             self.state = 'target'
#         else:
#             self.state = 'no_target'
#         #self.state = self.task_state[self.idx] #te.hdf.root.task_msgs[:]

#     def create_feature_extractor(self):
#         '''
#         Create the feature extractor object
#         '''
#         self.extractor = extractor.ReplaySpikeCountsExtractor(te.hdf.root.task, 
#             source='spike_counts', units=self.decoder.units)

#     def get_cursor_location(self):
#         if self.idx % 1000 == 0: 
#             print self.idx

#         self.current_assist_level = te.hdf.root.task[self.idx]['assist_level'][0]
#         self._update_target_loc()
#         spike_obs = self.get_spike_counts()
        
#         self.call_decoder_output = self.call_decoder(spike_obs.astype(np.float64))
#         self.decoder_state[self.idx] = self.call_decoder_output
#         self.idx += 1


reload(bmi_recon_tasks)
gen = genfns.sim_target_seq_generator_multi(8, 1000)
self = task = bmi_recon_tasks.ContCLDARecon(te, n_iter, gen)
task.init()
error = task.calc_recon_error()

print("Max recon error", np.max(np.abs(error)))

# task = CLDAPPFReconstruction(gen)
# task.init()

# self = task

# batch_idx = 0
# while self.idx < n_iter:
#     st = time.time()
#     self.get_cursor_location()
#     #print time.time() - st

# cursor = te.hdf.root.task[:]['cursor']
# if cursor.dtype == np.float32:
#     error = cursor[:n_iter] - np.float32(self.decoder_state[:n_iter,0:3,-1])
# else:
#     error = cursor[:n_iter] - self.decoder_state[:n_iter,0:3,-1]
# print "Recon error", np.max(np.abs(error))
