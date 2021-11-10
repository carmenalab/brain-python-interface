#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
swig -python -outdir $DIR $DIR/control_comedi_swig.i 
gcc -O3 -c -fPIC $DIR/control_comedi_swig.c -lcomedi -lm -o $DIR/control_comedi_swig.o $(pkg-config --cflags --libs python3)
gcc -O3 -c -fPIC $DIR/control_comedi_swig_wrap.c -lcomedi -lm -o $DIR/control_comedi_swig_wrap.o $(pkg-config --cflags --libs python3)
gcc -O3 -shared $DIR/control_comedi_swig.o $DIR/control_comedi_swig_wrap.o -lcomedi -lm -o $DIR/_control_comedi_swig.so