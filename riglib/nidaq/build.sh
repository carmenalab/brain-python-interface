#!/bin/bash
swig -python pcidio.i
gcc -c -fPIC pcidio.c pcidio_wrap.c -lcomedi -lm -I/usr/include/python2.7/
ld -shared pcidio.o pcidio_wrap.o -o _pcidio.so