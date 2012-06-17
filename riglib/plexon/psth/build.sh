#!/bin/bash
swig -python psth.i
gcc -pthread -fno-strict-aliasing -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC -I/usr/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c psth_wrap.c -o psth_wrap.o
gcc -pthread -fno-strict-aliasing -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC -I/usr/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c psth.c -o psth.o
gcc -pthread -shared -Wl,-O3 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro psth_wrap.o psth.o -o _psth.so
