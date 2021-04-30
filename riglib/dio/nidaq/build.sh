#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
swig -python -outdir $DIR $DIR/pcidio.i 
#gcc -O3 -c -fPIC $DIR/pcidio.c $DIR/pcidio_wrap.c -lcomedi -lm -I/usr/include/python2.7/
gcc -O3 -c -fPIC $DIR/pcidio.c -lcomedi -lm -o $DIR/pcidio.o $(pkg-config --cflags --libs python3)
gcc -O3 -c -fPIC $DIR/pcidio_wrap.c -lcomedi -lm -o $DIR/pcidio_wrap.o $(pkg-config --cflags --libs python3)
gcc -O3 -shared $DIR/pcidio.o $DIR/pcidio_wrap.o -lcomedi -lm -o $DIR/_pcidio.so
