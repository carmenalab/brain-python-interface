#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# ezrange extension module
_psth = Extension("_psth",
                   ["psth.i","psth.c"],
                   include_dirs = [numpy_include],
                   define_macros = [('DEBUG', 1)],
                   )

# ezrange setup
setup(  name        = "psth generator",
        description = "Generates the PSTH from the raw buffer made by plexnet",
        author      = "James Gao",
        version     = "1.0",
        ext_modules = [_psth]
        )
