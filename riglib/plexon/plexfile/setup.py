#! /usr/bin/env python

# System imports
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# ezrange extension module
plex = Extension("plexfile",
                   ["plexon.pyx", 'plexfile.c', 'plexread.c', 'dataframe.c', 'inspect.c'],
#                   define_macros = [('DEBUG', None)],
#                   extra_compile_args=["-g"],
#                   extra_link_args=["-g"],
                   )

# ezrange setup
setup(  name        = "Plexfile reader",
        description = "",
        author      = "James Gao",
        version     = "1.0",
        cmdclass = {'build_ext': build_ext},
        ext_modules = [plex]
        )
