#!/usr/bin/env python
'''
Install the cython code required to open plexon files. 
'''
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

psth = Extension("plexon.psth",
    ['plexon/psth.pyx', 'plexon/cpsth/psth.c'],
    include_dirs= ['.', np.get_include(), 'plexon/', 'plexon/cpsth/'],
)

plexfile = Extension("plexon.plexfile",
    ['plexon/plexfile.pyx',
     'plexon/cplexfile/plexfile.c', 
     'plexon/cplexfile/plexread.c', 
     'plexon/cplexfile/dataframe.c', 
     'plexon/cplexfile/inspect.c', 
     'plexon/cpsth/psth.c'],
    include_dirs= [ '.',
        np.get_include(), 
        'plexon/',
        'plexon/cpsth/', 
        'plexon/cplexfile/'
    ],
)

setup(  name        = "Plexfile utilities",
        description = "Utilities for dealing with the Plexon neural streaming interface",
        author      = "James Gao",
        version     = "0.1.0",
        packages = ['plexon'],
        ext_modules = cythonize([psth, plexfile], include_path=['.', 'plexon/cython/', np.get_include()])
        )
