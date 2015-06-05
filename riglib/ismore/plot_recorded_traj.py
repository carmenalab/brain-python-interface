import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from riglib.ismore.common_state_lists import *
from utils.util_fns import norm_vec
from utils.constants import *


# parse command line arguments
parser = argparse.ArgumentParser(description='Plot a recorded reference \
    trajectory from the saved reference .pkl file.')
parser.add_argument('trial_type', help='E.g., Blue, Red, "Brown to Red", etc.')
args = parser.parse_args()
trial_type = args.trial_type

traj_file_ref = os.path.expandvars('$BMI3D/riglib/ismore/traj_reference_interp.pkl')


traj_ref = pickle.load(open(traj_file_ref, 'rb'))
aa_flag = 'armassist' in traj_ref[trial_type]
rh_flag = 'rehand' in traj_ref[trial_type]


if aa_flag:
    aa_ref = traj_ref[trial_type]['armassist']
    print "length of aa_ref:", len(aa_ref)
    print "length of aa_ref (secs):", aa_ref['ts'][aa_ref.index[-1]] - aa_ref['ts'][0]
    
if rh_flag:
    rh_ref = traj_ref[trial_type]['rehand']
    print "length of rh_ref:", len(rh_ref)
    print "length of rh_ref (secs):", rh_ref['ts'][rh_ref.index[-1]] - rh_ref['ts'][0]
    

############
# PLOTTING #
############

color_ref = 'red'
tight_layout_kwargs = {
    'pad':   0.5,
    'w_pad': 0.5,
    'h_pad': 0.5,
}

if aa_flag:
    fig = plt.figure()
    plt.title('xy trajectories')
    plt.plot(aa_ref['aa_px'], aa_ref['aa_py'], '-D', color=color_ref,  markersize=5)
    plt.plot(aa_ref['aa_px'][0], aa_ref['aa_py'][0], 'D', color='green', markersize=10)
    plt.tight_layout(**tight_layout_kwargs)

    fig = plt.figure()
    plt.title('psi (ArmAssist orientation) trajectory')
    plt.plot(rad_to_deg * aa_ref['aa_ppsi'], color=color_ref)
    plt.tight_layout(**tight_layout_kwargs)
    

if rh_flag:   
    fig = plt.figure()
    grid = (4, 1)
    rh_tvec = rh_ref['ts'] - rh_ref['ts'][0]
    for i, state in enumerate(rh_pos_states):
        ax = plt.subplot2grid(grid, (i, 0))
        if i == 0:
            ax.set_title('rehand angle trajectories')
        plt.plot(rh_tvec, rad_to_deg * rh_ref[state], color=color_ref)
    plt.tight_layout(**tight_layout_kwargs)

plt.show()
