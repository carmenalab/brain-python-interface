#!/usr/bin/python
"""
Deterimine how well the binning method is working
"""
import os
import numpy as np
import tables
from plexon import plexfile

hdf_data_basename = 'cart20130911_09.hdf'
hdf_data_file = os.path.join('/storage/rawdata/hdf', hdf_data_basename)

plx_data_basename = 'cart20130911_09.plx'
plx_data_file = os.path.join('/storage/plexon', plx_data_basename)

# open the files
plx = plexfile.openFile(plx_data_file)
hdf = tables.openFile(hdf_data_file)
