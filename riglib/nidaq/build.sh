#!/bin/bash
swig -python pcidio.i
gcc -c -fPIC pcidio.c pcidio_wrap.c -lcomedi -lm -I/usr/include/python2.7/
gcc -shared pcidio.o pcidio_wrap.o -lcomedi -lm -o _pcidio.so
