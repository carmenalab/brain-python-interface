#!/bin/bash
swig -python pcidio.i
gcc -O3 -c -fPIC pcidio.c pcidio_wrap.c -lcomedi -lm -I/usr/include/python2.7/
gcc -O3 -shared pcidio.o pcidio_wrap.o -lcomedi -lm -o _pcidio.so
