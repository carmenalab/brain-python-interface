#! /usr/bin/env python

# System imports
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

psth = Extension("plexon.psth",
    ['plexon/cython/psth.pyx', 'plexon/psth/psth.c'],
    include_dirs= ['.', np.get_include(), 'plexon/', 'plexon/cython/', 'plexon/psth/'],
    # define_macros = [('DEBUG', None)],
    # extra_compile_args=["-g"],
    # extra_link_args=["-g"],
)

plexfile = Extension("plexon.plexfile",
    ['plexon/cython/plexfile.c',
     'plexon/plexfile/plexfile.c', 
     'plexon/plexfile/plexread.c', 
     'plexon/plexfile/dataframe.c', 
     'plexon/plexfile/inspect.c', 
     'plexon/psth/psth.c'],
    include_dirs= [ '.',
        np.get_include(), 
        'plexon/',
        'plexon/cython/',
        'plexon/psth/', 
        'plexon/plexfile/'
    ],
    # define_macros = [('DEBUG', None)],
    # extra_compile_args=["-g"],
    # extra_link_args=["-g"],
)

setup(  name        = "Plexfile utilities",
        description = "Utilities for dealing with the Plexon neural streaming interface",
        author      = "James Gao",
        version     = "0.1.0",
        packages = ['plexon'],
        ext_modules = cythonize([psth, plexfile], include_dirs=['.', 'plexon/cython/'])
        )
