import argparse
import tables
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from riglib.ismore import settings
from riglib.ismore.safetygrid import SafetyGrid


if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('hdf_name', help='Name of input .hdf file.')
    parser.add_argument('pkl_name', help='Name of input .pkl file.')
    parser.add_argument('local_dist', type=int, help='Distance in cm. For \
        each saved position in the recorded data, the min/max psi and prono \
        angular values in neighboring squares with a range of +/- \
        local_dist cm in both the x and y directions are updated.')
    args = parser.parse_args()

    # load data from hdf file
    hdf = tables.openFile(args.hdf_name)
    aa_data = hdf.root.armassist[:]['data']
    rh_flag = 'rehand' in hdf.root
    if rh_flag:
        rh_data = hdf.root.rehand[:]['data']

    # load SafetyGrid object from the .pkl file
    safety_grid = pickle.load(open(args.pkl_name, 'rb'))

    # for each recorded psi angle, update the min/max psi values (if necessary)
    #   of all positions within a local_dist cm radius
    for i in range(len(aa_data)):
        pos = np.array([aa_data[i]['aa_px'], aa_data[i]['aa_py']])
        psi = aa_data[i]['aa_ppsi']
        safety_grid.update_minmax_psi(pos, psi, args.local_dist)

    if rh_flag:
        # since the ArmAssist and ReHand data are asynchronous, we need to 
        #   interpolate the ArmAssist data onto the times (in ts_interp) at 
        #   which ReHand data is saved, so that we know the corresponding 
        #   ArmAssist xy-position for each prono value
        ts    = aa_data[:]['ts_arrival']
        aa_px = aa_data[:]['data']['aa_px']
        aa_py = aa_data[:]['data']['aa_py']
        ts_interp = rh_data[:]['ts_arrival']
        interp_aa_px = interp1d(ts, aa_px)(ts_interp)
        interp_aa_py = interp1d(ts, aa_py)(ts_interp)

        # for each recorded psi angle, update the min/max prono values 
        #   (if necessary) of all positions within a local_dist cm radius
        for i in range(len(rh_data)):
            pos = np.array([interp_aa_px[i], interp_aa_py[i]])
            prono = rh_data[i]['rh_prono']
            safety_grid.update_minmax_prono(pos, prono, args.local_dist)

    safety_grid.plot_minmax_psi()
    safety_grid.plot_minmax_prono()

    print ''
    if safety_grid.is_psi_minmax_set():
        print 'Min/max psi values were set for all valid grid squares!'
    else:
        print 'Min/max psi values were NOT set for all valid grid squares!'
        print 'Try one of the following:'
        print '  1) recording calibration movements that cover more of the workspace'
        print '  2) increase the local_dist parameter'
    print ''

    print ''
    if safety_grid.is_prono_minmax_set():
        print 'Min/max prono values were set for all valid grid squares!'
    else:
        print 'Min/max prono values were NOT set for all valid grid squares!'
        print 'Try one of the following:'
        print '  1) recording calibration movements that cover more of the workspace'
        print '  2) increase the local_dist parameter'
    print ''

    # save the SafetyGrid to a .pkl file with the same name as the .hdf file
    output_pkl_name = os.path.join('/storage/calibration', 
                                   os.path.basename(args.hdf_name)[:-4] + '.pkl')
    pickle.dump(safety_grid, open(output_pkl_name, 'wb'))

    plt.show()
