'''Script for processing the data saved in an HDF file to define the outer 
safety boundary. Saves the resulting SafetyGrid object to a .pkl file.'''

import argparse
import numpy as np
import tables
import pickle
import matplotlib.pyplot as plt
import os

from ismore import settings
from ismore.safetygrid import SafetyGrid



if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('hdf_name', help='Name of input .hdf file.')
    args = parser.parse_args()

    # load armassist data from hdf file
    hdf = tables.openFile(args.hdf_name)
    aa_data = hdf.root.armassist[:]['data']

    # create a SafetyGrid object
    mat_size = settings.MAT_SIZE
    delta = 0.5  # size (length/width in cm) of each square in the SafetyGrid
    safety_grid = SafetyGrid(mat_size, delta)
    
    boundary_positions = np.array([aa_data[:]['aa_px'], aa_data[:]['aa_py']]).T
    safety_grid.set_valid_boundary(boundary_positions)
    
    interior_pos = [np.mean(aa_data[:]['aa_px']), np.mean(aa_data[:]['aa_py'])]
    safety_grid.mark_interior_as_valid(interior_pos)
    
    safety_grid.plot_valid_area()
    print 'Total valid area: %.2f cm^2' % safety_grid.calculate_valid_area()

    # save the SafetyGrid to a .pkl file with the same name as the .hdf file
    pkl_name = os.path.join('/storage/calibration', 
                            os.path.basename(args.hdf_name)[:-4] + '.pkl')
    pickle.dump(safety_grid, open(pkl_name, 'wb'))

    plt.show()
