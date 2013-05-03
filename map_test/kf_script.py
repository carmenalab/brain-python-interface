import numpy as np
import pickle
import socket
import struct
import time
from scipy.io import loadmat

from riglib.bmi import kfdecoder
import dsp

sim = True 
T_loop = 0.1
n_iter = 100
run_clda = False

# Link to task display
display_ip_addr = '127.0.0.1' if sim else '192.168.0.2'
display_addr = (display_ip_addr, 22301)
display_soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

if sim: # Instantiate a simulation source
    from data_source_sim_spikes import DataSourceSimSpikes
    plx_data = DataSourceSimSpikes()
    display_soc.sendto('reset', display_addr)
else: # Instantiate a plexnet client object 
    from riglib import source
    from riglib.plexon import Spikes
    plx_data = source.DataSource(Spikes, addr=('10.0.0.13', 6000))
    #plx_data = source.DataSource(Spikes, addr=('192.168.0.6', 6000))
    plx_data.start()
    time.sleep(0.7) # Compensating for start-up delay?

# Instantiate low-pass filter
from scipy import signal
filtCutoff = 2.0
norm_pass = filtCutoff/(1./T_loop/2)
b, a = signal.butter(2, norm_pass, btype='low', analog=0, output='ba')
lpf = dsp.LTIFilter(b, a, ndim=2)

# Initialize loop data storage variables
slack = np.zeros(n_iter)
ts_data = [None]*n_iter

# Center-out task information
target_code_offset = 64
target_codes = np.arange(8) + target_code_offset

task_states = ['go_to_center', 'hold_at_center', 'go_to_target', 'hold_at_target']
trial_end_codes = [4, 8, 12, 9]
ENTERS_CENTER = 15
ENTERS_TARGET = 7
GO_CUE = 5

# Loat target locations
target_locations = loadmat('jeev_center_out_bmi_targets_post012813.mat')
center = target_locations['centerPos'].ravel()
targets = target_locations['targetPos']
horiz_min, vert_min = center - np.array([0.09, 0.09])
horiz_max, vert_max = center + np.array([0.09, 0.09])
bounding_box = (horiz_min, vert_min, horiz_max, vert_max)

# Load the decoder
if sim:
    decoder_fname = '/Users/sgowda/bmi/workspace/smoothbatch/jeev/test_decoder.mat'
    decoder = kfdecoder.load_from_mat_file(decoder_fname)
else:
    decoder_fname = 'jeev041513_VFB_Kawf_B100_NS5_NU14_Z1_smoothbatch_smoothbatch_smoothbatch_smoothbatch_smoothbatch_smoothbatch.mat'
    decoder = kfdecoder.load_from_mat_file(decoder_fname, bounding_box=bounding_box)
    
    # load the zscoring decoder
    norm_decoder_fname = 'jeev050113_VFB_Kawf_B100_NS5_NU14_Z1.mat'
    norm_decoder = kfdecoder.load_from_mat_file(norm_decoder_fname)
    decoder.init_zscore(norm_decoder.mFR, norm_decoder.sdFR)

if run_clda:
    CLDA_batchsize = 100
    #halflife = ??? # TODO look this up from the smoothbatch config file and/or spike_bmi_sim
    class CLDABatch():
        def __init__(self, batch_size):
            self.batch_size = batch_size
            self.kindata = []
            self.neuraldata = []
    
        def push(self, kindata, neuraldata):
            self.kindata.append(data)
            self.neuraldata.append(neuraldata)
    
        def is_full(self):
            return len(self.kindata) >= self.batch_size
    
        def get_batch(self):
            kindata = np.vstack(self.kindata).T
            self.kindata = []
            self.neuraldata = []
            return kindata, self.neuraldata
        
current_target = center
last_target_code = 64

assist_level = 0
for k in range(n_iter):
    # Get the time at the start of the loop
    t_loop_start = time.time()

    # Get spike timestamp data
    ts_data[k] = plx_data.get()

    # Decode binned spike counts using KF
    decoder.decode(ts_data[k], target=current_target, target_radius=0.012, 
        assist_level=assist_level)
    bmi_kin = decoder['p_x', 'p_y', 'v_x', 'v_y']

    # Low-pass filter the decoded cursor position
    #bmi_kin[0:2] = lpf(bmi_kin[0:2])

    # Send decoded position to task display
    bmi_output = struct.pack('d'*4, *bmi_kin)
    bmi_output += struct.pack('d', 0) # fake value for update flag
    display_soc.sendto(bmi_output, display_addr)

    task_events = ts_data[k][ts_data[k]['chan'] == 257, :]
    for event in task_events:
        event_code = event['unit']
        if event_code in target_codes: # assuming that 2 and target code appear simultaneously, i.e. normal center-out task
            current_task_state = 'go_to_center'
            current_target = center
            last_target_code = event_code
        elif event_code == ENTERS_CENTER:
            current_task_state = 'hold_at_center'
        elif event_code == ENTERS_TARGET:
            current_task_state = 'hold_at_target'
        elif event_code == GO_CUE:
            current_task_state = 'go_to_target'
            current_target = targets[last_target_code - target_code_offset, :]

    ##--- CLDA
    # Task events
    if run_clda:
        bmi_pos = bmi_kin[0:2]
        bmi_vel = bmi_kin[2:4]
        if not clda_batch.is_full():
            if 'hold' in current_task_state:
                reaimed_vel = np.zeros(2)
            else:
                intended_dir = current_target - bmi_pos
                norm_intended_dir = intended_dir/np.linalg.norm(intended_dir)
                reaimed_vel = max(np.linalg.norm(bmi_vel), 0.01) * norm_intended_dir

            cursor_goal_kin = np.hstack([bmi_pos, reaimed_vel]).reshape(-1,1)
            clda_batch.push(current_target, cursor_goal_kin)
        else:
            # Retrain!
            batch = clda_batch.get_batch()
            kfdecoder.retrain(batch, halflife)
       
    # Sleep off the remaining time until the next decoder output
    #slack[k] = (time.time() - t_loop_start)
    slack[k] = T_loop - (time.time() - t_loop_start)
    print slack[k]
    try:
        time.sleep(T_loop - (time.time() - t_loop_start))
    except: # slack is negative
        print "not sleeping!"

print "BMI ended (finished fixed # of iterations)"
if sim: # Terminate simulation source
    pass
else: # Instantiate a plexnet client object 
    plx_data.stop()
